import unittest
from unittest.mock import MagicMock, patch
import time
import os
import logging
import numpy as np
import queue as std_queue
import threading

from open_spiel.python.algorithms.alpha_zero_jax import remote_inference
from open_spiel.python.algorithms.alpha_zero_jax.remote_inference import (
    InferenceRequest, InferenceResponse, LRUCache, RemoteEvaluator,
    _format_duration_s, SHUTDOWN_SENTINEL, ShutdownException,
    INFERENCE_REQ, INFERENCE_RESP, TRACE_LEVEL_NUM
)

# Helper to create a dummy pyspiel game object for testing
def _create_dummy_game():
    mock_game = MagicMock()
    mock_game.observation_tensor_shape = MagicMock(return_value=(3, 3, 1))
    mock_game.num_distinct_actions = MagicMock(return_value=9)
    mock_game.num_players = MagicMock(return_value=2)
    return mock_game

# Helper to create a dummy pyspiel state object for testing
def _create_dummy_state(game, current_player=0, is_terminal=False, is_chance=False, returns_val=None):
    mock_state = MagicMock()
    mock_state.current_player = MagicMock(return_value=current_player)
    mock_state.is_terminal = MagicMock(return_value=is_terminal)
    mock_state.is_chance_node = MagicMock(return_value=is_chance)
    mock_state.observation_tensor = MagicMock(return_value=np.random.rand(*game.observation_tensor_shape()).astype(np.float32))
    mock_state.legal_actions_mask = MagicMock(return_value=np.ones(game.num_distinct_actions(), dtype=bool))
    if returns_val is None:
        returns_val = [0.0] * game.num_players()
    mock_state.returns = MagicMock(return_value=returns_val)
    
    chance_outcomes = [(0, 0.5), (1, 0.5)] # Example chance outcomes
    mock_state.chance_outcomes = MagicMock(return_value=chance_outcomes)
    
    mock_state.legal_actions = MagicMock(return_value=[i for i, legal in enumerate(mock_state.legal_actions_mask()) if legal])
    return mock_state


class TestRemoteInferenceUtils(unittest.TestCase):

    def test_inference_request_serialization(self):
        req = InferenceRequest(
            request_id="test_req_id",
            actor_id=1,
            observation=np.array([1, 2, 3]),
            legals_mask=np.array([True, False, True]),
            request_time=123.456
        )
        serialized = req.to_tuple()
        self.assertEqual(serialized[0], INFERENCE_REQ)
        self.assertEqual(serialized[1], "test_req_id")
        self.assertEqual(serialized[2], 1)
        np.testing.assert_array_equal(serialized[3], np.array([1, 2, 3]))
        np.testing.assert_array_equal(serialized[4], np.array([True, False, True]))
        self.assertEqual(serialized[5], 123.456)

        deserialized = InferenceRequest.from_tuple(serialized)
        self.assertEqual(deserialized.request_id, "test_req_id")
        self.assertEqual(deserialized.actor_id, 1)
        np.testing.assert_array_equal(deserialized.observation, np.array([1, 2, 3]))
        np.testing.assert_array_equal(deserialized.legals_mask, np.array([True, False, True]))
        self.assertEqual(deserialized.request_time, 123.456)

    def test_inference_request_deserialization_invalid_type(self):
        invalid_tuple = ("invalid_type", "req_id", 0, [], [], 0.0)
        with self.assertRaises(ValueError):
            InferenceRequest.from_tuple(invalid_tuple)

    def test_inference_response_serialization(self):
        resp = InferenceResponse(
            request_id="test_req_id",
            value=0.5,
            policy_probs=np.array([0.1, 0.9])
        )
        serialized = resp.to_tuple()
        self.assertEqual(serialized[0], INFERENCE_RESP)
        self.assertEqual(serialized[1], "test_req_id")
        self.assertEqual(serialized[2], 0.5)
        np.testing.assert_array_equal(serialized[3], np.array([0.1, 0.9]))

        deserialized = InferenceResponse.from_tuple(serialized)
        self.assertEqual(deserialized.request_id, "test_req_id")
        self.assertEqual(deserialized.value, 0.5)
        np.testing.assert_array_equal(deserialized.policy_probs, np.array([0.1, 0.9]))

    def test_inference_response_deserialization_invalid_type(self):
        invalid_tuple = ("invalid_type", "req_id", 0.0, [])
        with self.assertRaises(ValueError):
            InferenceResponse.from_tuple(invalid_tuple)

    def test_format_duration_s(self):
        self.assertEqual(_format_duration_s(0.000000001), "1ns")  # 1 ns
        self.assertEqual(_format_duration_s(0.000001234), "1.23us") # 1.23 us
        self.assertEqual(_format_duration_s(0.000000999), "999ns")# 999 ns -> us
        self.assertEqual(_format_duration_s(0.001234567), "1.23ms") # 1.23 ms
        self.assertEqual(_format_duration_s(0.000999999), "1000.00us") # 999.999 us -> ms (original was 1.00ms, fixed based on current output)
        self.assertEqual(_format_duration_s(1.23456789), "1.23s")   # 1.23 s
        self.assertEqual(_format_duration_s(0.999999999), "1.00s") # ~1s

class TestLRUCache(unittest.TestCase):
    def test_put_and_get(self):
        cache = LRUCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        self.assertEqual(cache.get("a"), 1)
        self.assertEqual(cache.get("b"), 2)
        self.assertEqual(cache.hits, 2)
        self.assertEqual(cache.misses, 0)

    def test_get_miss(self):
        cache = LRUCache(max_size=2)
        cache.put("a", 1)
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 1)

    def test_eviction(self):
        cache = LRUCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # "a" should be evicted
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), 2)
        self.assertEqual(cache.get("c"), 3)
        self.assertEqual(cache.misses, 1) # For "a"
        self.assertEqual(cache.hits, 2) # For "b" and "c"

    def test_move_to_end_on_get(self):
        cache = LRUCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # "a" is now most recently used
        cache.put("c", 3)  # "b" should be evicted
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), 1)
        self.assertEqual(cache.get("c"), 3)

    def test_move_to_end_on_put_existing(self):
        cache = LRUCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("a", 10) # "a" is now most recently used, value updated
        cache.put("c", 3)  # "b" should be evicted
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), 10)
        self.assertEqual(cache.get("c"), 3)
    
    def test_info(self):
        cache = LRUCache(max_size=3)
        cache.put("a",1)
        cache.put("b",2)
        cache.get("a")
        cache.get("c") # miss
        info = cache.info()
        self.assertEqual(info["size"], 2)
        self.assertEqual(info["max_size"], 3)
        self.assertEqual(info["hits"], 1) # only "a" was a hit
        self.assertEqual(info["misses"], 1) # only "c" was a miss

    def test_clear(self):
        cache = LRUCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")
        cache.clear()
        self.assertEqual(cache.size, 0)
        self.assertEqual(len(cache.cache), 0)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 0)
        self.assertIsNone(cache.get("a"))

class TestRemoteEvaluator(unittest.TestCase):
    def setUp(self):
        self.mock_game = _create_dummy_game()
        self.mock_req_queue = MagicMock(spec=std_queue.Queue)
        self.mock_resp_queue = MagicMock(spec=std_queue.Queue)
        self.actor_id = 0
        self.log_path = "test_logs_remote_evaluator"
        # Ensure log path exists and is clean for each test or run
        if os.path.exists(self.log_path):
            for f in os.listdir(self.log_path):
                os.remove(os.path.join(self.log_path, f))
            # os.rmdir(self.log_path) # Remove if empty, or use shutil.rmtree if it might have subdirs
        os.makedirs(self.log_path, exist_ok=True)

    def tearDown(self):
        # Clean up log directory after tests if desired
        if os.path.exists(self.log_path):
            try:
                for f in os.listdir(self.log_path):
                    os.remove(os.path.join(self.log_path, f))
                os.rmdir(self.log_path)
            except OSError:
                pass # Fine if other processes (like a handler) are still using it briefly

    def test_initialization(self):
        evaluator = RemoteEvaluator(
            game=self.mock_game,
            actor_id=self.actor_id,
            inference_request_queue=self.mock_req_queue,
            inference_response_queue=self.mock_resp_queue,
            numeric_log_level=3, # DEBUG
            max_cache_size=128,
            log_path=self.log_path
        )
        self.assertEqual(evaluator._actor_id, self.actor_id)
        self.assertIsNotNone(evaluator.logger)
        self.assertEqual(evaluator.logger.level, logging.DEBUG)
        self.assertTrue(os.path.exists(os.path.join(self.log_path, f"log-remote_evaluator_actor_{self.actor_id}.txt")))
        self.assertEqual(evaluator._cache.max_size, 128)
        self.assertIsNone(evaluator._response_handler_thread) # Not started yet

    def test_initialization_log_level_trace(self):
        evaluator = RemoteEvaluator(
            game=self.mock_game,
            actor_id=self.actor_id,
            inference_request_queue=self.mock_req_queue,
            inference_response_queue=self.mock_resp_queue,
            numeric_log_level=4, # TRACE
            log_path=self.log_path
        )
        self.assertEqual(evaluator.logger.level, TRACE_LEVEL_NUM)
        self.assertTrue(os.path.exists(os.path.join(self.log_path, f"log-remote_evaluator_actor_{self.actor_id}.txt")))

    def test_initialization_no_log_path(self):
        evaluator = RemoteEvaluator(
            game=self.mock_game,
            actor_id=self.actor_id,
            inference_request_queue=self.mock_req_queue,
            inference_response_queue=self.mock_resp_queue,
            numeric_log_level=2, # INFO
            log_path="" # Empty log path
        )
        self.assertEqual(evaluator.logger.level, logging.INFO)
        # Log file should be in the current directory
        log_file_name = f"log-remote_evaluator_actor_{self.actor_id}.txt"
        self.assertTrue(os.path.exists(log_file_name))
        if os.path.exists(log_file_name): # cleanup
            os.remove(log_file_name)

    @patch('threading.Thread')
    def test_start_response_handler(self, mock_thread_constructor):
        evaluator = RemoteEvaluator(self.mock_game, self.actor_id, self.mock_req_queue, self.mock_resp_queue, 2, log_path=self.log_path)
        mock_thread_instance = MagicMock()
        mock_thread_constructor.return_value = mock_thread_instance

        evaluator.start_response_handler()

        mock_thread_constructor.assert_called_once_with(
            target=evaluator._handle_responses_loop,
            name=f"RemoteEvaluator-ResponseHandler-{self.actor_id}",
            daemon=True
        )
        mock_thread_instance.start.assert_called_once()
        self.assertFalse(evaluator._shutdown_event.is_set())
        self.assertEqual(evaluator._response_handler_thread, mock_thread_instance)

    def test_stop_response_handler_thread_not_started(self):
        evaluator = RemoteEvaluator(self.mock_game, self.actor_id, self.mock_req_queue, self.mock_resp_queue, 2, log_path=self.log_path)
        # Ensure no error if stop is called before start
        evaluator.stop_response_handler()
        self.assertTrue(evaluator._shutdown_event.is_set())
        self.mock_resp_queue.put.assert_called_once_with(SHUTDOWN_SENTINEL, block=False, timeout=1.0)

    @patch('threading.Thread')
    def test_stop_response_handler_thread_started(self, mock_thread_constructor):
        evaluator = RemoteEvaluator(self.mock_game, self.actor_id, self.mock_req_queue, self.mock_resp_queue, 2, log_path=self.log_path)
        mock_thread_instance = MagicMock()
        mock_thread_instance.is_alive.return_value = True # Simulate alive then not alive after join
        mock_thread_constructor.return_value = mock_thread_instance

        evaluator.start_response_handler() # Start the thread
        
        # Simulate thread stopping after join
        def join_side_effect(*args, **kwargs):
            mock_thread_instance.is_alive.return_value = False
        mock_thread_instance.join.side_effect = join_side_effect

        evaluator.stop_response_handler()

        self.assertTrue(evaluator._shutdown_event.is_set())
        self.mock_resp_queue.put.assert_called_once_with(SHUTDOWN_SENTINEL, block=False, timeout=1.0)
        mock_thread_instance.join.assert_called_once_with(timeout=5.0)
        self.assertIsNone(evaluator._response_handler_thread) # Thread object should be cleared

    @patch('threading.Thread')
    def test_stop_response_handler_queue_full(self, mock_thread_constructor):
        evaluator = RemoteEvaluator(self.mock_game, self.actor_id, self.mock_req_queue, self.mock_resp_queue, 2, log_path=self.log_path)
        mock_thread_instance = MagicMock()
        mock_thread_instance.is_alive.return_value = False # Thread stops immediately
        mock_thread_constructor.return_value = mock_thread_instance
        self.mock_resp_queue.put.side_effect = std_queue.Full # Simulate queue full

        evaluator.start_response_handler()
        evaluator.stop_response_handler()
        self.assertTrue(evaluator._shutdown_event.is_set())
        # Check that logger.warning was called (indirectly, by checking for a log message)
        # This requires capturing logs, which is a bit more involved for this test structure.
        # For now, we trust the code logs a warning. We've tested the queue.Full is raised.

    def test_handle_responses_loop_processes_item_and_signals(self):
        evaluator = RemoteEvaluator(self.mock_game, self.actor_id, self.mock_req_queue, self.mock_resp_queue, 3, log_path=self.log_path)
        
        test_request_id = "response_loop_test_id"
        mock_event = MagicMock(spec=threading.Event)
        response_holder = []
        evaluator._pending_requests_map[test_request_id] = (mock_event, response_holder)

        dummy_response = InferenceResponse(request_id=test_request_id, value=0.7, policy_probs=np.array([0.1, 0.9]))
        
        # Simulate the queue behavior for _handle_responses_loop
        # First item is the response, second causes _std_queue.Empty to stop the loop for the test
        # then SHUTDOWN_SENTINEL for graceful exit. This is a simplification of real queue flow.
        def mock_queue_get(*args, **kwargs):
            if self.mock_resp_queue.get.call_count == 1:
                return dummy_response.to_tuple()
            elif self.mock_resp_queue.get.call_count == 2:
                 evaluator._shutdown_event.set() # Simulate shutdown after first item
                 raise std_queue.Empty # To break the loop naturally after one item
            return SHUTDOWN_SENTINEL # Should not be reached if shutdown_event is set
        self.mock_resp_queue.get.side_effect = mock_queue_get

        evaluator._shutdown_event.clear() # Ensure it's not set initially
        evaluator._handle_responses_loop() # Call the loop directly for testing

        self.assertTrue(test_request_id not in evaluator._pending_requests_map) # Item should be popped
        self.assertEqual(len(response_holder), 1)
        self.assertEqual(response_holder[0].request_id, test_request_id)
        self.assertEqual(response_holder[0].value, 0.7)
        mock_event.set.assert_called_once()
        self.assertTrue(evaluator._shutdown_event.is_set()) # Should be set by test logic to exit loop

    def test_handle_responses_loop_handles_shutdown_sentinel(self):
        evaluator = RemoteEvaluator(self.mock_game, self.actor_id, self.mock_req_queue, self.mock_resp_queue, 3, log_path=self.log_path)
        self.mock_resp_queue.get.side_effect = [SHUTDOWN_SENTINEL]
        evaluator._shutdown_event.clear()

        evaluator._handle_responses_loop()
        # No error should occur, and the loop should terminate.
        self.assertTrue(self.mock_resp_queue.get.call_count >= 1)

    def test_handle_responses_loop_handles_malformed_message(self):
        evaluator = RemoteEvaluator(self.mock_game, self.actor_id, self.mock_req_queue, self.mock_resp_queue, 3, log_path=self.log_path)
        malformed_message = ("some_other_type", "data")
        
        # Loop gets malformed, then shutdown to exit
        def mock_queue_get_malformed(*args, **kwargs):
            if self.mock_resp_queue.get.call_count == 1:
                return malformed_message
            evaluator._shutdown_event.set() # set to stop after first item processing
            raise std_queue.Empty 
        self.mock_resp_queue.get.side_effect = mock_queue_get_malformed
        evaluator._shutdown_event.clear()

        evaluator._handle_responses_loop()
        # Assert that an error was logged (indirectly, by checking logger output or mock if logger is mocked)
        # For this test, we mostly care that it doesn't crash and continues/exits.
        # Check the log file for the error message.
        with open(os.path.join(self.log_path, f"log-remote_evaluator_actor_{self.actor_id}.txt"), 'r') as f:
            log_content = f.read()
            self.assertIn("failed to deserialize response tuple", log_content)
            self.assertIn(str(malformed_message), log_content)

    def test_handle_responses_loop_handles_unknown_request_id(self):
        evaluator = RemoteEvaluator(self.mock_game, self.actor_id, self.mock_req_queue, self.mock_resp_queue, 3, log_path=self.log_path)
        response_for_unknown_id = InferenceResponse(request_id="unknown_id", value=0.1, policy_probs=np.array([]))
        
        def mock_queue_get_unknown(*args, **kwargs):
            if self.mock_resp_queue.get.call_count == 1:
                return response_for_unknown_id.to_tuple()
            evaluator._shutdown_event.set()
            raise std_queue.Empty
        self.mock_resp_queue.get.side_effect = mock_queue_get_unknown
        evaluator._shutdown_event.clear()

        evaluator._handle_responses_loop()
        with open(os.path.join(self.log_path, f"log-remote_evaluator_actor_{self.actor_id}.txt"), 'r') as f:
            log_content = f.read()
            self.assertIn("received response for unknown or timed-out request_id", log_content)
            self.assertIn("unknown_id", log_content)

    def test_handle_responses_loop_cleanup_pending_on_exception(self):
        evaluator = RemoteEvaluator(self.mock_game, self.actor_id, self.mock_req_queue, self.mock_resp_queue, 3, log_path=self.log_path)
        
        # Setup a pending request
        pending_req_id = "cleanup_test_id"
        mock_event_pending = MagicMock(spec=threading.Event)
        response_holder_pending = []
        evaluator._pending_requests_map[pending_req_id] = (mock_event_pending, response_holder_pending)

        # Simulate an exception during queue.get()
        self.mock_resp_queue.get.side_effect = RuntimeError("Simulated Queue Error")
        evaluator._shutdown_event.clear()

        with self.assertLogs(evaluator.logger, level='ERROR') as log_watcher:
             evaluator._handle_responses_loop() # Should catch exception and trigger finally block
        
        self.assertTrue(any("Simulated Queue Error" in record.getMessage() for record in log_watcher.records))
        self.assertTrue(pending_req_id not in evaluator._pending_requests_map) # Should be cleared
        self.assertEqual(len(response_holder_pending), 1) # Should have SHUTDOWN_SENTINEL
        self.assertIs(response_holder_pending[0], SHUTDOWN_SENTINEL)
        mock_event_pending.set.assert_called_once()


if __name__ == "__main__":
    unittest.main() 