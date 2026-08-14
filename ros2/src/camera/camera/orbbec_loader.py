import glob
import importlib
import os
import sys
import sysconfig


def _available_orbbec_extensions(lib_dir):
    available = ', '.join(
        os.path.basename(path)
        for path in glob.glob(os.path.join(lib_dir, 'orbbec*.so'))
    )
    return available or '无'


class _PyorbbecDeviceInfo:
    def __init__(self, info):
        self._info = info

    def get_sn(self):
        return self._info.get_serial_number()

    def get_firmware_version(self):
        return self._info.get_firmware_version()

    def __getattr__(self, name):
        return getattr(self._info, name)


class _PyorbbecOB:
    def __init__(self, sdk):
        self._sdk = sdk
        self._pipeline = None
        self._camera_param = None
        self._pending_frames = None
        self._pending_color_frame = None
        self._pending_depth_frame = None
        self._color_consumed = True
        self._depth_consumed = True

    def startPipeline(self, device):
        self._pipeline = self._sdk.Pipeline(device) if device is not None else self._sdk.Pipeline()

        errors = []
        for align_mode in (self._sdk.OBAlignMode.HW_MODE, self._sdk.OBAlignMode.SW_MODE):
            config = self._build_config(align_mode)
            try:
                try:
                    self._pipeline.enable_frame_sync()
                except Exception:
                    pass

                self._pipeline.start(config)
                self._prime_camera_param()
                return True
            except Exception as exc:
                errors.append(f'{align_mode.name}: {exc}')
                try:
                    self._pipeline.stop()
                except Exception:
                    pass

        raise RuntimeError('启动 pyorbbecsdk Pipeline 失败: ' + '; '.join(errors))

    def stopPipeline(self):
        if self._pipeline is not None:
            self._pipeline.stop()

    def getDepthFrame(self):
        self._ensure_pending_frames()
        frame = self._pending_depth_frame
        self._depth_consumed = True
        self._clear_if_consumed()
        return self._depth_frame_to_mm(frame)

    def getColorFrame(self):
        self._ensure_pending_frames()
        frame = self._pending_color_frame
        self._color_consumed = True
        self._clear_if_consumed()
        return self._color_frame_to_rgb(frame)

    def get3DPoint(self, x, y, depth_mm):
        if self._camera_param is None:
            return None

        intrinsic = self._camera_param.rgb_intrinsic
        if not intrinsic.fx or not intrinsic.fy:
            intrinsic = self._camera_param.depth_intrinsic

        z = float(depth_mm)
        point_x = (float(x) - intrinsic.cx) * z / intrinsic.fx
        point_y = (float(y) - intrinsic.cy) * z / intrinsic.fy
        return point_x, point_y, z

    def _build_config(self, align_mode):
        config = self._sdk.Config()
        missing = []

        for sensor_type in (self._sdk.OBSensorType.DEPTH_SENSOR, self._sdk.OBSensorType.COLOR_SENSOR):
            try:
                profile_list = self._pipeline.get_stream_profile_list(sensor_type)
                profile = profile_list.get_default_video_stream_profile()
                config.enable_stream(profile)
            except Exception as exc:
                missing.append(f'{sensor_type.name}: {exc}')

        if missing:
            raise RuntimeError('无法启用 Orbbec 深度/彩色流: ' + '; '.join(missing))

        config.set_align_mode(align_mode)
        config.set_frame_aggregate_output_mode(self._sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        return config

    def _prime_camera_param(self):
        for _ in range(30):
            frames = self._pipeline.wait_for_frames(1000)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue

            self._set_pending_frames(frames, color_frame, depth_frame)
            self._camera_param = self._pipeline.get_camera_param()
            return

        raise RuntimeError('无法从 Orbbec 相机获取首帧或相机内参')

    def _ensure_pending_frames(self):
        if self._pending_frames is not None:
            return

        for _ in range(5):
            frames = self._pipeline.wait_for_frames(1000)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue

            self._set_pending_frames(frames, color_frame, depth_frame)
            return

        raise RuntimeError('等待 Orbbec 彩色/深度帧超时')

    def _set_pending_frames(self, frames, color_frame, depth_frame):
        self._pending_frames = frames
        self._pending_color_frame = color_frame
        self._pending_depth_frame = depth_frame
        self._color_consumed = False
        self._depth_consumed = False

    def _clear_if_consumed(self):
        if self._color_consumed and self._depth_consumed:
            self._pending_frames = None
            self._pending_color_frame = None
            self._pending_depth_frame = None

    def _depth_frame_to_mm(self, frame):
        import numpy as np

        width = frame.get_width()
        height = frame.get_height()
        scale = frame.get_depth_scale()
        data = np.asarray(frame.get_data())

        if data.dtype == np.uint16:
            depth = data.reshape((height, width))
        else:
            depth = np.frombuffer(data, dtype=np.uint16).reshape((height, width))

        return depth.astype(np.float32) * float(scale)

    def _color_frame_to_rgb(self, frame):
        import cv2
        import numpy as np

        width = frame.get_width()
        height = frame.get_height()
        data = np.asarray(frame.get_data(), dtype=np.uint8)
        color_format = frame.get_format()

        if color_format == self._sdk.OBFormat.RGB:
            return data.reshape((height, width, 3)).copy()
        if color_format == self._sdk.OBFormat.BGR:
            image = data.reshape((height, width, 3))
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if color_format in (self._sdk.OBFormat.YUYV, self._sdk.OBFormat.YUY2):
            image = data.reshape((height, width, 2))
            return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_YUY2)
        if color_format == self._sdk.OBFormat.UYVY:
            image = data.reshape((height, width, 2))
            return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_UYVY)
        if color_format == self._sdk.OBFormat.MJPG:
            bgr = cv2.imdecode(data.reshape(-1), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError('MJPG 彩色帧解码失败')
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if color_format == self._sdk.OBFormat.I420:
            image = data.reshape((height * 3 // 2, width))
            return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_I420)
        if color_format == self._sdk.OBFormat.NV12:
            image = data.reshape((height * 3 // 2, width))
            return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_NV12)
        if color_format == self._sdk.OBFormat.NV21:
            image = data.reshape((height * 3 // 2, width))
            return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_NV21)

        return self._convert_color_with_sdk(frame)

    def _convert_color_with_sdk(self, frame):
        import numpy as np

        mapping = {
            self._sdk.OBFormat.YUYV: self._sdk.OBConvertFormat.YUYV_TO_RGB888,
            self._sdk.OBFormat.YUY2: self._sdk.OBConvertFormat.YUYV_TO_RGB888,
            self._sdk.OBFormat.I420: self._sdk.OBConvertFormat.I420_TO_RGB888,
            self._sdk.OBFormat.NV21: self._sdk.OBConvertFormat.NV21_TO_RGB888,
            self._sdk.OBFormat.NV12: self._sdk.OBConvertFormat.NV12_TO_RGB888,
            self._sdk.OBFormat.MJPG: self._sdk.OBConvertFormat.MJPG_TO_RGB888,
            self._sdk.OBFormat.UYVY: self._sdk.OBConvertFormat.UYVY_TO_RGB888,
            self._sdk.OBFormat.BGR: self._sdk.OBConvertFormat.BGR_TO_RGB,
        }
        convert_format = mapping.get(frame.get_format())
        if convert_format is None:
            raise RuntimeError(f'不支持的 Orbbec 彩色帧格式: {frame.get_format()}')

        convert_filter = self._sdk.FormatConvertFilter()
        convert_filter.set_format_convert_format(convert_format)
        rgb_frame = convert_filter.process(frame)
        if rgb_frame is None:
            raise RuntimeError(f'Orbbec 彩色帧格式转换失败: {frame.get_format()}')

        width = rgb_frame.get_width()
        height = rgb_frame.get_height()
        data = np.asarray(rgb_frame.get_data(), dtype=np.uint8)
        return data.reshape((height, width, 3)).copy()


class _PyorbbecCompatModule:
    def __init__(self, sdk):
        self._sdk = sdk

    def ob_init(self):
        context = self._sdk.Context()
        device_list = context.query_devices()
        if device_list.get_count() <= 0:
            raise RuntimeError('未检测到 Orbbec 相机，请确认 USB 连接和 udev 权限')

        device = device_list.get_device_by_index(0)
        device_info = _PyorbbecDeviceInfo(device.get_device_info())
        return True, context, device, device_info

    def OB(self):
        return _PyorbbecOB(self._sdk)

    def __getattr__(self, name):
        return getattr(self._sdk, name)


def load_orbbec():
    lib_dir = os.path.join(os.path.dirname(__file__), 'lib')
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)

    try:
        return importlib.import_module('orbbec')
    except ImportError:
        try:
            pyorbbecsdk = importlib.import_module('pyorbbecsdk')
            return _PyorbbecCompatModule(pyorbbecsdk)
        except ImportError as pyorbbec_exc:
            extension_suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
            expected = os.path.join(lib_dir, f'orbbec*{extension_suffix}')
            available = _available_orbbec_extensions(lib_dir)
            raise ImportError(
                '未找到可用的 Orbbec Python 绑定。'
                f'当前 Python 需要 {os.path.basename(expected)}，已发现旧扩展: {available}。'
                '也未能导入 Python 3.12 版 pyorbbecsdk。'
                '请安装 pyorbbecsdk2，或放入 cpython-312/abi3 版本的 orbbec .so。'
            ) from pyorbbec_exc
