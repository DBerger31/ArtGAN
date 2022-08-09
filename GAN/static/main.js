function show_image(src, width, height, alt) {
    var img = document.createElement("img");
    img.src = src;
    img.width = width;
    img.height = height;
    img.alt = alt;

    document.body.appendChild(img);
}

function show_multi_image(width, height, alt) {
    var row = document.createElement("div");
    row.classList.add('row')
    row.style.cssText =  'display: grid; grid-auto-flow: column; gap: 1px; align-items: center; justify-items: center;';
    row.id = 'row-imgs'
    // document.body.appendChild(row)
    for(var i=0; i <10; i++){

        var col = document.createElement("div");
        col.id = 'col-md-8'
        // col.style.cssText =  'display: grid; grid-auto-flow: column; gap: 4px; align-items: center; justify-items: center;';

        var img = document.createElement("img");
        img.src = '/static/img' + i + '.png?' + new Date().getTime();
        img.width = width;
        img.height = height;
        img.alt = alt;
        
        col.appendChild(img)
        row.appendChild(col)
        // document.body.appendChild(col)
        // document.body.appendChild(img);
    }
    document.body.appendChild(row)

}

const btn = document.getElementById("test")
function disable_button() {
    btn.disabled = true;
    setTimeout(()=>{
    btn.disabled = false;
    console.log('Button Activated')}, 5000)
}
