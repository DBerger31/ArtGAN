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
    // row.style.cssText = 'position:absolute;top:300px;left:1600px;width:200px;height:200px;-moz-border-radius:100px;border:1px  solid #ddd;-moz-box-shadow: 0px 0px 8px  #fff;display:none;';
    document.body.appendChild(row)
    for(var i=0; i <10; i++){

        var col = document.createElement("div");
        col.classList.add('col-md-8')
        col.style.cssText =  'display:none;';
        var img = document.createElement("img");
        img.src = '/static/img' + i + '.png?' + new Date().getTime();
        img.width = width;
        img.height = height;
        img.alt = alt;
        
        document.body.appendChild(col)
        document.body.appendChild(img);
    }

}
