
document.addEventListener('DOMContentLoaded', function() {


    const image = document.getElementById("xray_image");
    const fileInput = document.querySelector("input[type=file]");
        fileInput.addEventListener("change", function(e) {
            image.src = '/static/images/scan_image.png';
            const uploadFileParent = this.closest(".uploadFile");
            const filenameElement = uploadFileParent.querySelector(".filename");
            filenameElement.textContent = e.target.files[0].name;
    });

    var myDropdown = document.querySelector('.my-dropdown');
    myDropdown.addEventListener('change', function() {
        var selectedValue = this.value;
        console.log(selectedValue)
        fetch('/diseaseprediction/?selected_value=' + selectedValue)
        .then(response => response.json())
        .then(data => {
            console.log(data,'data')
        });
    });

});
