import unittest
from unittest.mock import patch, MagicMock
import sys

# Force mock dependencies
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['google.genai.types'] = MagicMock()
sys.modules['google.genai.chats'] = MagicMock()
sys.modules['pydantic'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.io'] = MagicMock()
sys.modules['dotenv'] = MagicMock()
sys.modules['adi'] = MagicMock()

from app.chatbot.chatbot import Chatbot

class TestChatbot(unittest.TestCase):

    def setUp(self):
        from app.chatbot.chatbot import gemini_client
        gemini_client.chats.create.reset_mock()
        gemini_client.chats.create.return_value = MagicMock()

    def test_init(self):
        from app.chatbot.chatbot import gemini_client
        
        bot = Chatbot()
        
        self.assertIsNotNone(bot.chat)
        self.assertEqual(bot.model, "gemini-2.5-flash-lite")
        gemini_client.chats.create.assert_called_once()

    def test_change_model(self):
        from app.chatbot.chatbot import gemini_client
        
        bot = Chatbot()
        
        # Mock get_history
        bot.chat.get_history.return_value = ["history"]
        
        bot.change_model("new-model")
        
        self.assertEqual(bot.model, "new-model")
        # Should be called twice: once for init, once for change_model
        self.assertEqual(gemini_client.chats.create.call_count, 2)
        
        # Verify the second call used the new model and history
        call_args = gemini_client.chats.create.call_args_list[1]
        self.assertEqual(call_args.kwargs['model'], "new-model")
        self.assertEqual(call_args.kwargs['history'], ["history"])

    def test_clear_history(self):
        from app.chatbot.chatbot import gemini_client
        
        bot = Chatbot()
        bot.clear_history()
        
        # Should be called twice
        self.assertEqual(gemini_client.chats.create.call_count, 2)
        
        # Verify the second call used empty history
        call_args = gemini_client.chats.create.call_args_list[1]
        self.assertEqual(call_args.kwargs['history'], [])

    def test_send_message_with_context(self):
        from app.chatbot.chatbot import gemini_client, types
        
        bot = Chatbot()
        
        bot.send_message_with_context("hello", include_graphs=[("graph1", b"bytes")])
        
        # Verify types.Part.from_bytes was called
        types.Part.from_bytes.assert_called_with(data=b"bytes", mime_type='image/png')

if __name__ == '__main__':
    unittest.main()
