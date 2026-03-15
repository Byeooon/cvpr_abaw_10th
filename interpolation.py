import os

def generate_forward_fill_submission():
    # 1. Path configuration
    base_dir = os.path.dirname(__file__)
    example_txt_path = "/media/SSD/data/CVPR_workshop/test_release/test_set_examples/test_set_examples 2/ABAW_Expr_test_set_example.txt"
    my_pred_path = os.path.join(base_dir, "predictions.txt")
    final_output_path = os.path.join(base_dir, "final_predictions.txt")

    print("=" * 60)
    print("[Final Submission Generation] Restore missing frames using the previous frame's prediction.")
    print("=" * 60)

    # 2. Load my prediction results into a dictionary
    predictions_map = {}
    if not os.path.exists(my_pred_path):
        print(f"[Error] Could not find my prediction file: {my_pred_path}")
        return

    print("1. Loading model inference results into memory...")
    with open(my_pred_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            if line.strip():
                parts = line.strip().split(',')
                img_loc = parts[0]
                pred_class = parts[1]
                predictions_map[img_loc] = pred_class

    print(f" -> Successfully loaded predictions: {len(predictions_map):,}")

    # 3. Use the official example file as a template to build the final file
    if not os.path.exists(example_txt_path):
        print(f"[Error] Could not find the official example file: {example_txt_path}")
        return

    print(f"2. Writing the final file ({final_output_path})...")
    match_count = 0
    missing_count = 0
    
    with open(example_txt_path, 'r') as template_f, open(final_output_path, 'w') as out_f:
        lines = template_f.readlines()
        
        # Write the header line
        out_f.write(lines[0])
        
        last_seen_value = '0'  # Tracks the previous prediction value (default: Neutral)
        current_video = ""     # Detects when the video changes
        
        # Match lines sequentially and apply forward fill
        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split(',')
                img_loc = parts[0]
                video_name = img_loc.split('/')[0]
                
                # Reset to the default value (0) when a new video starts
                if video_name != current_video:
                    current_video = video_name
                    last_seen_value = '0' 
                
                if img_loc in predictions_map:
                    # Use the model prediction if available and store it for the next frame
                    final_pred = predictions_map[img_loc]
                    last_seen_value = final_pred
                    match_count += 1
                else:
                    # If the frame is missing, copy the most recent stored value
                    final_pred = last_seen_value
                    missing_count += 1
                    
                out_f.write(f"{img_loc},{final_pred}\n")

    print("-" * 60)
    print("[Done] Final submission file generation is complete.")
    print(f" - Existing model predictions used: {match_count:,}")
    print(f" - Missing frames restored (copied previous value): {missing_count:,}")
    print(f" - Total lines (excluding header): {match_count + missing_count:,}")
    print("=" * 60)

if __name__ == "__main__":
    generate_forward_fill_submission()
