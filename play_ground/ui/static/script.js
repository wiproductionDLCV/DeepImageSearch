let selectedGround = false;
let selectedSynthetic = false;

function checkFolderSelections() {
    if (selectedGround && selectedSynthetic) {
        document.getElementById("compareSection").style.display = "block";
    }
}

function selectFolder(type) {
    fetch('/select-folder')
        .then(res => res.json())
        .then(data => {
            const path = data.folderPath || "No folder selected.";
            if (type === 'ground') {
                document.getElementById("groundPath").innerText = path;
                selectedGround = true;
            } else if (type === 'synthetic') {
                document.getElementById("syntheticPath").innerText = path;
                selectedSynthetic = true;
            } else if (type === 'visualize') {
                document.getElementById("visualizePath").innerText = path;
            }
            checkFolderSelections();
        });
}

document.getElementById("compareBtn").addEventListener("click", () => {
    document.getElementById("compareStatus").innerText = "Comparing images...";
    
    setTimeout(() => {
        document.getElementById("compareStatus").innerText = "✅ Comparison completed!";
        document.getElementById("visualizeSection").style.display = "block";
    }, 4000); // Simulate 4 sec comparison
});