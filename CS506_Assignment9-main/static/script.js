document.getElementById("experiment-form").addEventListener("submit", async function(event) {
    event.preventDefault();  // Prevent form submission

    const activation = document.getElementById("activation").value;
    const lr = parseFloat(document.getElementById("lr").value);
    const stepNum = parseInt(document.getElementById("step_num").value);

    // Validation checks
    const acts = ["relu", "tanh", "sigmoid"];
    if (!acts.includes(activation)) {
        alert("Please choose from relu, tanh, sigmoid.");
        return;
    }

    if (isNaN(lr)) {
        alert("Please enter a valid number for learning rate.");
        return;
    }

    if (isNaN(stepNum) || stepNum <= 0) {
        alert("Please enter a positive integer for Number of Training Steps.");
        return;
    }

    // Disable the button to prevent duplicate submissions
    const submitButton = document.querySelector('button[type="submit"]');
    submitButton.disabled = true;

    // If all validations pass, submit the form
    try {
        const response = await fetch("/run_experiment", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ activation: activation, lr: lr, step_num: stepNum })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Show and set images if they exist
        const resultsDiv = document.getElementById("results");
        resultsDiv.style.display = "block";

        const resultImg = document.getElementById("result_gif");
        if (data.result_gif) {
            resultImg.src = `/${data.result_gif}`;
            resultImg.style.display = "block";
        }
    } catch (error) {
        console.error("Error running experiment:", error);
        alert("An error occurred while running the experiment.");
    } finally {
        // Re-enable the button
        submitButton.disabled = false;
    }
});
