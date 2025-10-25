document.addEventListener("DOMContentLoaded", function () {
    const frontPage = document.querySelector(".front-page");
    const backgroundImage = document.querySelector(".background-image");
    const knowMoreContainer = document.querySelector(".know-more-container");
    const loginButtonContainer = document.querySelector(".login-button-container");
    const translucentSlide = document.getElementById("translucent-slide");
    const loginButton = document.getElementById("login-button");
    const cancelButton = document.getElementById("cancel-button");

    // Fade out front page after 2 seconds
    setTimeout(() => {
        frontPage.classList.add("fade-out");
        backgroundImage.classList.add("show");
        setTimeout(() => {
            frontPage.style.display = "none"; // Hide front page after fade-out
        }, 500); // Match this duration with the fade-out transition
    }, 2000);

    // Move the "know more" text after 2.5 seconds
    setTimeout(() => {
        knowMoreContainer.classList.add("shrink");
        loginButtonContainer.style.opacity = 1; // Show the login button
    }, 2500);

    // Show translucent slide on login button click
    loginButton.addEventListener("click", () => {
        translucentSlide.classList.add("show");
    });

    // Hide translucent slide on cancel button click
    cancelButton.addEventListener("click", () => {
        translucentSlide.classList.remove("show");
    });
});