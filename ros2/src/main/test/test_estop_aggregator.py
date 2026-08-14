from __future__ import annotations

import unittest

from main.estop_aggregator import MultiSourceEstopState, parse_estop_request


class EstopAggregatorStateTest(unittest.TestCase):
    def test_parse_accepts_valid_payload(self) -> None:
        request, error = parse_estop_request(
            '{"source":"voice_assistant","active":true}'
        )

        self.assertIsNone(error)
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.source, 'voice_assistant')
        self.assertTrue(request.active)
        self.assertTrue(request.latch)

    def test_parse_accepts_release_with_reset_sources_extension(self) -> None:
        request, error = parse_estop_request(
            '{"source":"voice_assistant","active":false,"latch":false,'
            '"reset_sources":["min_distance_camera"]}'
        )

        self.assertIsNone(error)
        self.assertIsNotNone(request)
        assert request is not None
        self.assertFalse(request.active)
        self.assertFalse(request.latch)

    def test_parse_rejects_invalid_payloads(self) -> None:
        for payload in (
            'not-json',
            '[]',
            '{"source":"","active":true}',
            '{"source":"voice","active":"true"}',
            '{"source":"voice","active":true,"latch":"yes"}',
        ):
            with self.subTest(payload=payload):
                request, error = parse_estop_request(payload)
                self.assertIsNone(request)
                self.assertIsNotNone(error)

    def test_multiple_sources_hold_stop_until_all_clear(self) -> None:
        state = MultiSourceEstopState()

        self.assertTrue(state.apply(_request('voice_assistant', True, True)))
        self.assertTrue(state.apply(_request('web_ui', True, True)))
        self.assertTrue(state.apply(_request('voice_assistant', False, False)))
        self.assertEqual(state.active_sources, ('web_ui',))
        self.assertFalse(state.apply(_request('web_ui', False, False)))

    def test_latched_source_requires_unlatched_release(self) -> None:
        state = MultiSourceEstopState()

        self.assertTrue(state.apply(_request('voice_assistant', True, True)))
        self.assertTrue(state.apply(_request('voice_assistant', False, True)))
        self.assertEqual(state.active_sources, ('voice_assistant',))
        self.assertFalse(
            state.apply(_request('voice_assistant', False, False))
        )


def _request(source: str, active: bool, latch: bool):
    request, error = parse_estop_request(
        f'{{"source":"{source}",'
        f'"active":{str(active).lower()},'
        f'"latch":{str(latch).lower()}}}'
    )
    assert error is None
    assert request is not None
    return request


if __name__ == '__main__':
    unittest.main()
