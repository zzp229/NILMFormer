"""
Script to display training results including MAE, MSE, and TECA metrics
"""
import torch
import pandas as pd
import json


def display_results(result_path):
    """
    Load and display training results from .pt file

    Args:
        result_path: Path to the .pt result file
    """
    print(f"Loading results from: {result_path}")
    print("=" * 80)

    # Load the results
    results = torch.load(result_path, map_location='cpu', weights_only=False)

    print("\n📊 Available keys in results:")
    for key in results.keys():
        print(f"  - {key}")

    # Display training history
    if 'loss_train_history' in results and 'loss_valid_history' in results:
        print("\n" + "=" * 80)
        print("📈 Training History:")
        print("=" * 80)
        train_loss = results['loss_train_history']
        valid_loss = results['loss_valid_history']

        for epoch, (t_loss, v_loss) in enumerate(zip(train_loss, valid_loss), 1):
            print(f"Epoch {epoch:3d}  |  Train Loss: {t_loss:.6f}  |  Valid Loss: {v_loss:.6f}")

    # Display test metrics
    if 'test_metrics_timestamp' in results:
        print("\n" + "=" * 80)
        print("🎯 Test Set Metrics (Timestamp-level):")
        print("=" * 80)

        test_metrics = results['test_metrics_timestamp']

        if isinstance(test_metrics, dict):
            for metric_name, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{metric_name:30s}: {value:.6f}")
                else:
                    print(f"{metric_name:30s}: {value}")
        else:
            print(test_metrics)

    # Display test window metrics
    if 'test_metrics_win' in results:
        print("\n" + "=" * 80)
        print("🎯 Test Set Metrics (Window-level):")
        print("=" * 80)

        test_metrics = results['test_metrics_win']

        if isinstance(test_metrics, dict):
            for metric_name, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{metric_name:30s}: {value:.6f}")
                else:
                    print(f"{metric_name:30s}: {value}")
        else:
            print(test_metrics)

    # Display validation metrics
    if 'valid_metrics_timestamp' in results:
        print("\n" + "=" * 80)
        print("✅ Validation Set Metrics (Timestamp-level):")
        print("=" * 80)

        valid_metrics = results['valid_metrics_timestamp']

        if isinstance(valid_metrics, dict):
            for metric_name, value in valid_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{metric_name:30s}: {value:.6f}")
                else:
                    print(f"{metric_name:30s}: {value}")
        else:
            print(valid_metrics)

    # Display validation window metrics
    if 'valid_metrics_win' in results:
        print("\n" + "=" * 80)
        print("✅ Validation Set Metrics (Window-level):")
        print("=" * 80)

        valid_metrics = results['valid_metrics_win']

        if isinstance(valid_metrics, dict):
            for metric_name, value in valid_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{metric_name:30s}: {value:.6f}")
                else:
                    print(f"{metric_name:30s}: {value}")
        else:
            print(valid_metrics)

    # Display aggregated metrics (MAE, MSE, TECA at different time scales)
    for freq in ['D', 'W', 'ME']:
        key = f'test_metrics_{freq}'
        if key in results:
            print("\n" + "=" * 80)
            print(f"📅 Test Metrics Aggregated by {freq} (Day/Week/Month):")
            print("=" * 80)

            metrics = results[key]
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"{metric_name:30s}: {value:.6f}")
                    else:
                        print(f"{metric_name:30s}: {value}")

    # Display predictions if available
    if 'test_outputs' in results:
        print("\n" + "=" * 80)
        print("🔮 Prediction Statistics:")
        print("=" * 80)
        outputs = results['test_outputs']
        if isinstance(outputs, dict):
            if 'y_true' in outputs and 'y_pred' in outputs:
                y_true = outputs['y_true']
                y_pred = outputs['y_pred']
                print(f"Number of predictions: {len(y_pred)}")
                print(f"True values - Mean: {y_true.mean():.2f}, Std: {y_true.std():.2f}, Min: {y_true.min():.2f}, Max: {y_true.max():.2f}")
                print(f"Pred values - Mean: {y_pred.mean():.2f}, Std: {y_pred.std():.2f}, Min: {y_pred.min():.2f}, Max: {y_pred.max():.2f}")

    print("\n" + "=" * 80)
    print("✨ Results display completed!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        result_file = sys.argv[1]
    else:
        # Default path based on the training command
        result_file = "result/UKDALE_WashingMachine_1min/128/NILMFormer_0.pt"

    try:
        results = display_results(result_file)
    except FileNotFoundError:
        print(f"Error: Result file not found at {result_file}")
        print("\nUsage: python show_result.py [path_to_result.pt]")
        print(f"Example: python show_result.py result/UKDALE_WashingMachine_1min/128/NILMFormer_0.pt")
    except Exception as e:
        print(f"Error loading results: {e}")
        import traceback
        traceback.print_exc()
