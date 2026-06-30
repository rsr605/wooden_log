/* ============================================================
   Wooden Log Detection — Frontend JavaScript
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const fileInfo = document.getElementById("file-info");
    const fileName = document.getElementById("file-name");
    const btnClear = document.getElementById("btn-clear");
    const detectForm = document.getElementById("detect-form");
    const btnDetect = document.getElementById("btn-detect");
    const confidenceSlider = document.getElementById("confidence");
    const confidenceValue = document.getElementById("confidence-value");

    // --- Click to open file dialog ---
    if (dropzone) {
        dropzone.addEventListener("click", () => fileInput.click());
    }

    // --- Drag & drop ---
    if (dropzone) {
        ["dragenter", "dragover"].forEach((evt) => {
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach((evt) => {
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove("dragover");
            });
        });

        dropzone.addEventListener("drop", (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                showFileInfo(files[0]);
            }
        });
    }

    // --- File selection ---
    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                showFileInfo(e.target.files[0]);
            }
        });
    }

    // --- Show file info ---
    function showFileInfo(file) {
        if (!file) return;

        // Validate file type
        const validExtensions = [
            "jpg", "jpeg", "png", "bmp", "webp",
            "mp4", "avi", "mov", "mkv"
        ];
        const ext = file.name.split(".").pop().toLowerCase();
        if (!validExtensions.includes(ext)) {
            alert("Unsupported file type: ." + ext);
            fileInput.value = "";
            return;
        }

        // Validate file size (50 MB)
        const maxSize = 50 * 1024 * 1024;
        if (file.size > maxSize) {
            alert("File too large. Maximum size is 50 MB.");
            fileInput.value = "";
            return;
        }

        // Display
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        fileName.textContent = `${file.name} (${sizeMB} MB)`;
        fileInfo.style.display = "flex";
    }

    // --- Clear file ---
    if (btnClear) {
        btnClear.addEventListener("click", () => {
            fileInput.value = "";
            fileInfo.style.display = "none";
        });
    }

    // --- Confidence slider ---
    if (confidenceSlider && confidenceValue) {
        confidenceSlider.addEventListener("input", (e) => {
            confidenceValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    // --- Form submit with loading state ---
    if (detectForm) {
        detectForm.addEventListener("submit", (e) => {
            if (!fileInput.files || fileInput.files.length === 0) {
                e.preventDefault();
                alert("Please select a file first.");
                return;
            }

            // Disable button and show spinner
            btnDetect.disabled = true;
            btnDetect.querySelector(".btn-text").textContent = "Processing...";

            // Create loading overlay
            const overlay = document.createElement("div");
            overlay.className = "loading-overlay active";
            overlay.innerHTML = `
                <div class="spinner">
                    <div class="spinner-circle"></div>
                    <div class="spinner-text">Detecting wooden logs...</div>
                </div>
            `;
            document.body.appendChild(overlay);
        });
    }
});
