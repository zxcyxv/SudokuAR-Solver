import torch
from models.transformer import SudokuTransformer

def verify():
    print("Verifying AR-TRM Baseline Model...")
    
    # Init options
    model = SudokuTransformer(num_layers=2, hidden_dim=384, num_heads=6)
    model.eval()
    
    # Dummy input
    # Batch size 2, Sequence Length 81
    bs = 2
    seq_len = 82 # 1 SOS + 81 Cells? Or just 81. Let's assume dataset gives 82.
    # Dataset produces Inputs (SOS->last) Length 81.
    # (Since total 82, input len 81)
    
    pos_in = torch.randint(0, 81, (bs, 81))
    val_in = torch.randint(0, 11, (bs, 81))
    
    print(f"Input Shapes: Pos {pos_in.shape}, Val {val_in.shape}")
    
    try:
        logits_pos, logits_val = model(pos_in, val_in)
        print(f"Output Shapes: Pos {logits_pos.shape}, Val {logits_val.shape}")
        
        # Check shapes
        assert logits_pos.shape == (bs, 81, 81)
        assert logits_val.shape == (bs, 81, 10)
        
        print("PASSED: verification success.")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
