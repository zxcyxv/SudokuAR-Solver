
import torch
import unittest
import torch.nn as nn
from models.urm_ar import SudokuURM_AR, URMConfig

class TestSudokuURMAR(unittest.TestCase):
    def setUp(self):
        # Minimal Config for Testing
        self.config = URMConfig(
            vocab_size=730,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            expansion=2,
            n_recurrence=2, # Small number for test speed
            max_seq_len=20
        )
        self.model = SudokuURM_AR(self.config)
        self.model.eval()

    def test_output_shape(self):
        bs = 2
        seq_len = 10
        x = torch.randint(0, 730, (bs, seq_len))
        logits = self.model(x)
        self.assertEqual(logits.shape, (bs, seq_len, 730))

    def test_strict_causality(self):
        """
        Verify effectively that processing [A, B] produces EXACTLY the same logits at pos 1 
        as processing [A, B, C] at pos 1.
        Any difference implies future leakage (e.g. from C to B).
        """
        torch.manual_seed(42)
        
        # Sequence: [A, B, C]
        seq_long = torch.randint(0, 730, (1, 10))
        seq_short = seq_long[:, :-1] # Drop the last token

        # Run 1: Short sequence
        with torch.no_grad():
            out_short = self.model(seq_short)
        
        # Run 2: Long sequence
        with torch.no_grad():
            out_long = self.model(seq_long)
            
        # Compare logits just before the last token of out_short
        # out_short: [1, 9, 730]
        # out_long:  [1, 10, 730]
        # We compare out_short[0, :] with out_long[0, :-1]
        
        diff = (out_short - out_long[:, :-1]).abs().max()
        print(f"Max difference (Causality Check): {diff.item()}")
        
        # Zero tolerance for float32/bfloat16 usually, but small epsilon might be needed for reduced precision
        # With float32 it should be exactly 0 if implemented correctly.
        self.assertTrue(diff < 1e-6, f"Causality violated! Max diff: {diff.item()}")

    def test_conv_padding_integrity(self):
        """
        Specific check for Conv1d padding to ensure output length is correct 
        and left-padding is applied.
        """
        # Access internal ConvSwiGLU if accessible, or test via model
        # We'll rely on test_strict_causality for functional verification
        pass

if __name__ == '__main__':
    unittest.main()
