# A sample 'Golden Dataset' for your automotive agent
eval_samples = [
    {
        "question": "How do I reset the oil life on a 2024 Ford F-150?",
        "ground_truth": "Navigate to the 'Vehicle' menu on the instrument cluster, select 'Oil Life', and hold the 'OK' button until it resets to 100%.",
        "reference_context": "The 2024 F-150 manual states: 'To reset oil life, use the steering wheel controls to find Settings > Vehicle > Oil Life Reset and hold OK.'"
    },
    {
        "question": "What does the solid red battery light mean?",
        "ground_truth": "A solid red battery light indicates a fault in the charging system, meaning the battery is not being charged.",
        "reference_context": "Dashboard lights section: 'Red Battery Icon: Charging system failure. The vehicle is running on battery power alone.'"
    }
]