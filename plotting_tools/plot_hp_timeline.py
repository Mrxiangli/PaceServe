import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to hp_timeline.csv (e.g., exp_result/.../replica_0/hp_timeline.csv)')
    parser.add_argument('--out', type=str, default='exp_result/images/hp_timeline.pdf', help='Output PDF path')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Log file {csv_path} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    if df.empty:
        print("No events found in log.")
        sys.exit(0)

    # Normalize time to start from 0
    t0 = df['Time'].min()
    df['time_rel'] = df['Time'] - t0

    plt.figure(figsize=(14, 6))

    # Plot tickets
    tickets = df[df['Event'] == 'TICKET_GENERATED']
    if not tickets.empty:
        plt.scatter(tickets['time_rel'], [1]*len(tickets), marker='^', color='#9B59B6', s=80, label='Ticket Generated')

    # Plot stolen
    stolen = df[df['Event'] == 'STOLEN_RECEIVED']
    if not stolen.empty:
        stolen_nonzero = stolen[stolen['Count'] > 0]
        stolen_zero = stolen[stolen['Count'] == 0]
        if not stolen_nonzero.empty:
            plt.scatter(stolen_nonzero['time_rel'], [2]*len(stolen_nonzero), marker='o', color='#70A5D9', s=80, label='Stolen Received (>0)')
        if not stolen_zero.empty:
            plt.scatter(stolen_zero['time_rel'], [2]*len(stolen_zero), marker='x', color='red', s=80, label='Stolen Failed (0)')

    # Plot urgent
    urgent = df[df['Event'] == 'URGENT_RECEIVED']
    if not urgent.empty:
        urgent_nonzero = urgent[urgent['Count'] > 0]
        urgent_zero = urgent[urgent['Count'] == 0]
        if not urgent_nonzero.empty:
            plt.scatter(urgent_nonzero['time_rel'], [3]*len(urgent_nonzero), marker='s', color='#ECA680', s=120, label='Urgent Received (>0)')
        if not urgent_zero.empty:
            plt.scatter(urgent_zero['time_rel'], [3]*len(urgent_zero), marker='x', color='darkred', s=100, label='Urgent Failed (0)')

    plt.yticks([1, 2, 3], ['Tickets', 'Stolen Requests', 'Urgent Requests'], fontsize=12)
    plt.xlabel('Time (seconds from first event)', fontsize=14)
    plt.title('HP Instance Timeline of Events', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    
    # Save directory
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Timeline plot saved to {out_path}")
    print("\n--- Event Counts ---")
    print(df['Event'].value_counts().to_string())

if __name__ == '__main__':
    main()
