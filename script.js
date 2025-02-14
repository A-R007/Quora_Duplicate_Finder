function checkDuplicate() {
    let question1 = document.getElementById("question1").value;
    let question2 = document.getElementById("question2").value;
    let resultDiv = document.getElementById("result");

    fetch("/check_duplicate", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ question1: question1, question2: question2 })
    })
    .then(response => response.json())
    .then(data => {
        resultDiv.innerHTML = data.result;
        resultDiv.className = "result-bubble " + (data.result === "Duplicate" ? "duplicate" : "not-duplicate");

        // Trigger shimmer animation
        resultDiv.style.opacity = "1";
        resultDiv.style.animation = "shimmer 1s ease-in-out";
    })
    .catch(error => console.error("Error:", error));
}
