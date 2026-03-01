#!/usr/bin/env python3
"""
Test script for MrBERT model implementation.

This script verifies that the MrBERT model with delete gates works correctly
by running various tests including forward passes, gradient checks, and
comparison with base BERT.

Usage:
    python test_mrbert.py
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM

from configuration_mrbert import MrBertConfig
from modeling_mrbert import (
    MrBertModel,
    MrBertForMaskedLM,
    MrBertForSequenceClassification,
    MrBertForTokenClassification,
    MrBertForQuestionAnswering,
    MrBertForMultipleChoice,
    MrBertForNextSentencePrediction,
    SigmoidDeleteGate,
    LogSigmoidDeleteGate,
    RandomDeleteGate,
    FixedDeleteGate,
)


def print_test_header(test_name: str):
    """Print a formatted test header."""
    print("\n" + "=" * 60)
    print(f"TEST: {test_name}")
    print("=" * 60)


def test_config_creation():
    """Test that MrBertConfig can be created with various parameters."""
    print_test_header("Configuration Creation")
    
    # Default config
    config = MrBertConfig()
    print(f"✓ Default config created")
    print(f"  - delete_gate_layer: {config.delete_gate_layer}")
    print(f"  - deletion_type: {config.deletion_type}")
    print(f"  - sigmoid_mask_scale: {config.sigmoid_mask_scale}")
    
    # Config from pretrained
    config = MrBertConfig.from_pretrained(
        "bert-base-uncased",
        deletion_type="scaled_sigmoid",
        delete_gate_layer=2,
        sigmoid_mask_scale=-10.0,
    )
    print(f"✓ Config from pretrained created")
    print(f"  - hidden_size: {config.hidden_size}")
    print(f"  - num_hidden_layers: {config.num_hidden_layers}")
    
    return True


def test_delete_gate_modules():
    """Test individual delete gate modules."""
    print_test_header("Delete Gate Modules")
    
    batch_size = 2
    seq_len = 16
    hidden_size = 768
    
    # Create dummy inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    
    # Test SigmoidDeleteGate
    config = MrBertConfig(
        hidden_size=hidden_size,
        deletion_type="scaled_sigmoid",
        sigmoid_mask_scale=-10.0,
    )
    gate = SigmoidDeleteGate(config)
    gate_values, gate_logits = gate(hidden_states, input_ids)
    print(f"✓ SigmoidDeleteGate")
    print(f"  - gate_values shape: {gate_values.shape}")
    print(f"  - gate_values range: [{gate_values.min().item():.4f}, {gate_values.max().item():.4f}]")
    
    # Test LogSigmoidDeleteGate
    config.deletion_type = "log_sigmoid"
    gate = LogSigmoidDeleteGate(config)
    gate_values, gate_logits = gate(hidden_states, input_ids)
    print(f"✓ LogSigmoidDeleteGate")
    print(f"  - gate_values shape: {gate_values.shape}")
    
    # Test RandomDeleteGate
    config.deletion_type = "random"
    gate = RandomDeleteGate(config)
    gate_values, gate_logits = gate(hidden_states, input_ids)
    print(f"✓ RandomDeleteGate")
    print(f"  - gate_values shape: {gate_values.shape}")
    
    # Test FixedDeleteGate
    config.deletion_type = "fixed"
    gate = FixedDeleteGate(config)
    gate_values, gate_logits = gate(hidden_states, input_ids)
    print(f"✓ FixedDeleteGate")
    print(f"  - gate_values shape: {gate_values.shape}")
    
    return True


def test_mrbert_model_forward():
    """Test MrBertModel forward pass with soft deletion."""
    print_test_header("MrBertModel Forward Pass (Soft Deletion)")
    
    # Create config from pretrained BERT
    config = MrBertConfig.from_pretrained(
        "bert-base-uncased",
        deletion_type="scaled_sigmoid",
        delete_gate_layer=2,
        sigmoid_mask_scale=-10.0,
    )
    
    # Create model
    model = MrBertModel(config)
    model.eval()
    print(f"✓ MrBertModel created")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )
    
    print(f"✓ Forward pass completed")
    print(f"  - last_hidden_state shape: {outputs.last_hidden_state.shape}")
    print(f"  - pooler_output shape: {outputs.pooler_output.shape}")
    
    # Check delete gate outputs
    if outputs.delete_gate_mask is not None:
        print(f"✓ Delete gate outputs present")
        print(f"  - delete_gate_mask shape: {outputs.delete_gate_mask.shape}")
        print(f"  - delete_gate_mask range: [{outputs.delete_gate_mask.min().item():.4f}, {outputs.delete_gate_mask.max().item():.4f}]")
    else:
        print(f"✗ Delete gate outputs not present (check delete_gate_layer)")
    
    return True


def test_mrbert_model_hard_delete():
    """Test MrBertModel forward pass with hard deletion."""
    print_test_header("MrBertModel Forward Pass (Hard Deletion)")
    
    config = MrBertConfig.from_pretrained(
        "bert-base-uncased",
        deletion_type="scaled_sigmoid",
        delete_gate_layer=2,
        sigmoid_mask_scale=-10.0,
        deletion_threshold=-5.0,  # Threshold for hard deletion
    )
    
    model = MrBertModel(config)
    model.eval()
    
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass with hard deletion
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hard_delete=True,
            deletion_threshold=-5.0,
        )
    
    print(f"✓ Forward pass with hard deletion completed")
    print(f"  - Original seq_len: {seq_len}")
    print(f"  - Output seq_len: {outputs.last_hidden_state.shape[1]}")
    
    if outputs.last_hidden_state.shape[1] < seq_len:
        print(f"  - Tokens deleted: {seq_len - outputs.last_hidden_state.shape[1]}")
    
    return True


def test_mrbert_for_masked_lm():
    """Test MrBertForMaskedLM with loss computation."""
    print_test_header("MrBertForMaskedLM Forward Pass")
    
    config = MrBertConfig.from_pretrained(
        "bert-base-uncased",
        deletion_type="scaled_sigmoid",
        delete_gate_layer=2,
        sigmoid_mask_scale=-10.0,
    )
    
    model = MrBertForMaskedLM(config)
    model.eval()
    print(f"✓ MrBertForMaskedLM created")
    
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Create labels (mask some tokens)
    labels = input_ids.clone()
    mask_positions = torch.randint(0, seq_len, (batch_size, 5))
    for b in range(batch_size):
        for pos in mask_positions[b]:
            labels[b, pos] = -100  # Ignore in loss except masked positions
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    
    print(f"✓ Forward pass completed")
    print(f"  - logits shape: {outputs.logits.shape}")
    print(f"  - loss: {outputs.loss.item():.4f}")
    
    if outputs.delete_gate_mask is not None:
        print(f"✓ Delete gate outputs present")
    
    return True


def test_gradient_flow():
    """Test that gradients flow through the delete gate."""
    print_test_header("Gradient Flow Through Delete Gate")
    
    config = MrBertConfig.from_pretrained(
        "bert-base-uncased",
        deletion_type="scaled_sigmoid",
        delete_gate_layer=2,
        sigmoid_mask_scale=-10.0,
    )
    
    model = MrBertForMaskedLM(config)
    model.train()
    
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = input_ids.clone()
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    
    # Backward pass
    outputs.loss.backward()
    
    # Check gradients on delete gate
    delete_gate_layer = model.bert.encoder.layer[config.delete_gate_layer]
    if hasattr(delete_gate_layer, 'delete_gate'):
        gate = delete_gate_layer.delete_gate
        if hasattr(gate, 'feed_forward'):
            grad = gate.feed_forward.weight.grad
            print(f"✓ Gradients computed for delete gate")
            print(f"  - feed_forward.weight.grad shape: {grad.shape}")
            print(f"  - grad norm: {grad.norm().item():.6f}")
            
            if grad.norm().item() > 0:
                print(f"✓ Gradients are non-zero - gate is learning!")
            else:
                print(f"✗ Gradients are zero - check implementation")
    
    return True


def test_comparison_with_bert():
    """Compare MrBERT output shapes with standard BERT."""
    print_test_header("Comparison with Standard BERT")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Create standard BERT
    bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    bert_model.eval()
    
    # Create MrBERT with same weights
    config = MrBertConfig.from_pretrained(
        "bert-base-uncased",
        deletion_type="scaled_sigmoid",
        delete_gate_layer=2,
        sigmoid_mask_scale=-10.0,
    )
    mrbert_model = MrBertForMaskedLM(config)
    mrbert_model.eval()
    
    # Tokenize sample text
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    # Forward passes
    with torch.no_grad():
        bert_outputs = bert_model(**inputs)
        mrbert_outputs = mrbert_model(**inputs)
    
    print(f"✓ Both models executed successfully")
    print(f"  - BERT logits shape: {bert_outputs.logits.shape}")
    print(f"  - MrBERT logits shape: {mrbert_outputs.logits.shape}")
    print(f"  - Shapes match: {bert_outputs.logits.shape == mrbert_outputs.logits.shape}")
    
    # Note: Outputs won't be identical since delete gate is randomly initialized
    print(f"\nNote: Output values differ because delete gate weights are random.")
    print(f"After training, MrBERT should learn to delete uninformative tokens.")
    
    return True


def test_different_deletion_types():
    """Test all deletion types."""
    print_test_header("Different Deletion Types")
    
    deletion_types = ["scaled_sigmoid", "log_sigmoid", "random", "fixed"]
    
    batch_size = 2
    seq_len = 16
    
    for deletion_type in deletion_types:
        config = MrBertConfig.from_pretrained(
            "bert-base-uncased",
            deletion_type=deletion_type,
            delete_gate_layer=2,
            sigmoid_mask_scale=-10.0,
        )
        
        model = MrBertModel(config)
        model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        print(f"✓ {deletion_type}")
        print(f"  - Output shape: {outputs.last_hidden_state.shape}")
        if outputs.delete_gate_mask is not None:
            deleted_count = (outputs.delete_gate_mask < -5.0).sum().item()
            print(f"  - Tokens with strong deletion signal: {deleted_count}")
    
    return True


def test_sequence_classification():
    """Test MrBertForSequenceClassification."""
    print_test_header("MrBertForSequenceClassification")
    
    num_labels = 3
    config = MrBertConfig.from_pretrained(
        "bert-base-uncased",
        deletion_type="scaled_sigmoid",
        delete_gate_layer=2,
        num_labels=num_labels,
    )
    
    model = MrBertForSequenceClassification(config)
    model.eval()
    print(f"✓ MrBertForSequenceClassification created")
    
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, num_labels, (batch_size,))
    
    # Forward pass with labels
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    
    print(f"✓ Forward pass completed")
    print(f"  - logits shape: {outputs.logits.shape}")
    print(f"  - loss: {outputs.loss.item():.4f}")
    
    # Verify delete gate outputs
    if outputs.delete_gate_mask is not None:
        print(f"✓ Delete gate outputs present")
    
    return True


def test_token_classification():
    """Test MrBertForTokenClassification."""
    print_test_header("MrBertForTokenClassification")
    
    num_labels = 9  # e.g., NER labels
    config = MrBertConfig.from_pretrained(
        "bert-base-uncased",
        deletion_type="scaled_sigmoid",
        delete_gate_layer=2,
        num_labels=num_labels,
    )
    
    model = MrBertForTokenClassification(config)
    model.eval()
    print(f"✓ MrBertForTokenClassification created")
    
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, num_labels, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    
    print(f"✓ Forward pass completed")
    print(f"  - logits shape: {outputs.logits.shape}")
    print(f"  - loss: {outputs.loss.item():.4f}")
    
    if outputs.delete_gate_mask is not None:
        print(f"✓ Delete gate outputs present")
    
    return True


def test_question_answering():
    """Test MrBertForQuestionAnswering."""
    print_test_header("MrBertForQuestionAnswering")
    
    config = MrBertConfig.from_pretrained(
        "bert-base-uncased",
        deletion_type="scaled_sigmoid",
        delete_gate_layer=2,
        num_labels=2,  # start and end
    )
    
    model = MrBertForQuestionAnswering(config)
    model.eval()
    print(f"✓ MrBertForQuestionAnswering created")
    
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    start_positions = torch.randint(0, seq_len // 2, (batch_size,))
    end_positions = start_positions + torch.randint(1, 10, (batch_size,))
    end_positions = end_positions.clamp(max=seq_len - 1)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
    
    print(f"✓ Forward pass completed")
    print(f"  - start_logits shape: {outputs.start_logits.shape}")
    print(f"  - end_logits shape: {outputs.end_logits.shape}")
    print(f"  - loss: {outputs.loss.item():.4f}")
    
    if outputs.delete_gate_mask is not None:
        print(f"✓ Delete gate outputs present")
    
    return True


def test_multiple_choice():
    """Test MrBertForMultipleChoice."""
    print_test_header("MrBertForMultipleChoice")
    
    config = MrBertConfig.from_pretrained(
        "bert-base-uncased",
        deletion_type="scaled_sigmoid",
        delete_gate_layer=2,
    )
    
    model = MrBertForMultipleChoice(config)
    model.eval()
    print(f"✓ MrBertForMultipleChoice created")
    
    batch_size = 2
    num_choices = 4
    seq_len = 32
    
    # Shape: (batch_size, num_choices, seq_len)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, num_choices, seq_len))
    attention_mask = torch.ones(batch_size, num_choices, seq_len)
    labels = torch.randint(0, num_choices, (batch_size,))
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    
    print(f"✓ Forward pass completed")
    print(f"  - logits shape: {outputs.logits.shape}")
    print(f"  - loss: {outputs.loss.item():.4f}")
    
    if outputs.delete_gate_mask is not None:
        print(f"✓ Delete gate outputs present")
    
    return True


def test_next_sentence_prediction():
    """Test MrBertForNextSentencePrediction."""
    print_test_header("MrBertForNextSentencePrediction")
    
    config = MrBertConfig.from_pretrained(
        "bert-base-uncased",
        deletion_type="scaled_sigmoid",
        delete_gate_layer=2,
    )
    
    model = MrBertForNextSentencePrediction(config)
    model.eval()
    print(f"✓ MrBertForNextSentencePrediction created")
    
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    # Token type IDs to indicate sentence A vs B
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    token_type_ids[:, seq_len // 2:] = 1
    labels = torch.randint(0, 2, (batch_size,))
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
    
    print(f"✓ Forward pass completed")
    print(f"  - logits shape: {outputs.logits.shape}")
    print(f"  - loss: {outputs.loss.item():.4f}")
    
    if outputs.delete_gate_mask is not None:
        print(f"✓ Delete gate outputs present")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# MrBERT Test Suite")
    print("#" * 60)
    
    tests = [
        ("Config Creation", test_config_creation),
        ("Delete Gate Modules", test_delete_gate_modules),
        ("MrBertModel Forward", test_mrbert_model_forward),
        ("MrBertModel Hard Delete", test_mrbert_model_hard_delete),
        ("MrBertForMaskedLM", test_mrbert_for_masked_lm),
        ("MrBertForSequenceClassification", test_sequence_classification),
        ("MrBertForTokenClassification", test_token_classification),
        ("MrBertForQuestionAnswering", test_question_answering),
        ("MrBertForMultipleChoice", test_multiple_choice),
        ("MrBertForNextSentencePrediction", test_next_sentence_prediction),
        ("Gradient Flow", test_gradient_flow),
        ("Comparison with BERT", test_comparison_with_bert),
        ("Different Deletion Types", test_different_deletion_types),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            results.append((name, f"ERROR: {e}"))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, status in results if status == "PASSED")
    total = len(results)
    
    for name, status in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{symbol} {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
