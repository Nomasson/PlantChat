
function upload_image(elem){

    document.cookie = `filesize=${elem.files[0].size}`;
    document.getElementById("exp").style.display = "none";
    document.getElementById("loader").style.display = "block";
    document.getElementById("wait").style.display = "block";
    setTimeout(function(){ document.getElementById('sub').click(); }, 3000);

    // document.getElementById('sub').click();



}

