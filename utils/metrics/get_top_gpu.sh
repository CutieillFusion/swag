ls ~/.. | while read user; do
  count=$(
    sacct -S 2024-09-01 -u "$user" --format=Elapsed 2>/dev/null \
      | tail -n +2 \
      | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }' \
      | paste -sd+ \
      | bc -l
  )
  if [ $? -eq 0 ]; then
    echo "$user: $count"
  fi
done | sort -n -k2 -t: -r | head -n 10