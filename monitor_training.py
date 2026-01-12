import time
import os

log_file = "training_fixed_delta.log"

print("Monitoring training progress...")
print("Press Ctrl+C to stop monitoring\n")

last_size = 0
last_epoch = 0

try:
    while True:
        # Check if file exists
        if os.path.exists(log_file):
            # Get current file size
            current_size = os.path.getsize(log_file)

            if current_size > last_size:
                # Read new content
                with open(log_file, 'r', encoding='utf-8') as f:
                    f.seek(last_size)
                    new_content = f.read()

                    # Extract and print epoch lines
                    for line in new_content.split('\n'):
                        if 'Epoch [' in line and '/100]' in line:
                            print(line.strip())
                        elif 'Early stopping' in line or 'Training complete' in line:
                            print(f"\n{'='*70}")
                            print(line.strip())
                            print('='*70)
                        elif 'MODEL COLLAPSE' in line:
                            pass  # Skip false alarm warnings

                last_size = current_size

        time.sleep(10)  # Check every 10 seconds

except KeyboardInterrupt:
    print("\n\nMonitoring stopped.")
