
import unittest
import torch
import torch.nn.functional as F
from models.urm_ar import SudokuURM_AR, URMConfig
from models.transformer_v3 import SOS_TOKEN, NUM_ACTIONS

class TestURMCache(unittest.TestCase):
    def setUp(self):
        self.config = URMConfig(
            vocab_size=730,
            hidden_size=64, 
            num_layers=1,
            num_heads=2,
            expansion=2,
            n_recurrence=1,
            max_seq_len=50
        )
        self.model = SudokuURM_AR(self.config)
        self.model.eval()

    def test_cache_vs_stateless(self):
        """
        Compare output of generate_fast (KV Cache) vs manual stateless generation.
        """
        torch.manual_seed(42)
        
        # Random "Board" (Just a sequence of tokens)
        seq_len = 10
        input_ids = torch.randint(0, 730, (1, seq_len))
        
        # 1. Stateless Forward (Ground Truth)
        with torch.no_grad():
            logits_stateless = self.model(input_ids, use_cache=False)
            last_logits_stateless = logits_stateless[0, -1]

        # 2. Stateful Forward (KV Cache)
        # Feed one by one
        past_caches = None
        with torch.no_grad():
            for i in range(seq_len):
                token = input_ids[:, i:i+1]
                logits_step, past_caches = self.model(token, use_cache=True, past_caches=past_caches)
            
            last_logits_cache = logits_step[0, -1]

        # Compare
        diff = (last_logits_stateless - last_logits_cache).abs().max()
        print(f"Max Difference (Stateless vs Cache): {diff.item()}")
        
        # Allow small epsilon for float precision
        self.assertTrue(diff < 1e-5, f"Cache implementation mismatch! Diff: {diff.item()}")

if __name__ == '__main__':
    unittest.main()
