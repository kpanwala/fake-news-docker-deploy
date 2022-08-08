const buildUrl = (url, parameters) => {
    var qs = "";
    for (var key in parameters) {
        var value = parameters[key];
        qs += encodeURIComponent(key) + "=" + encodeURIComponent(value) + "&";
    }
    if (qs.length > 0) {
        qs = qs.substring(0, qs.length - 1); //chop off last "&"
        url = url + "?" + qs;
    }
    return url;
}

document.querySelector('#predict').addEventListener('click', (e) => {
    e.preventDefault();

    // getting parameters necessary
    const algo = document.querySelector('#Algorithm').value;
    const granularity = document.querySelector('#Granularity').value;
    const text = btoa(unescape(encodeURIComponent(document.querySelector('#text').value)));


    // preparing url
    const baseurl = 'http://847aeb9545c0.ngrok.io';
    const parameters = {
        algo,
        granularity,
        text
    };
    const finalURL = buildUrl(baseurl, parameters);

    console.log(finalURL)
    let Http = new XMLHttpRequest();

    Http.open("GET", finalURL);
    Http.onreadystatechange = (e) => {
        if (Http.readyState === 4 && Http.status === 200) {
            console.log(Http.responseText)
            const response = JSON.parse(Http.responseText);
            let result;
            const real = "<div class=' p-3 mb-2 bg-success text-white rounded'>It's Real News.</div>";
            const fake = "<div class='p-3 mb-2 bg-danger text-white rounded'>It's Fake News.</div>";
            const err = "<div class='p-3 mb-2 bg-info text-white rounded'>oops! an error occured, try again!</div>"
            if (response === "True") {
                $('body').removeClass('Fakenewsbody');
                result = real;
                $('body').addClass('RealNewsbody')
            } else if (response === "Fake") {
                $('body').removeClass('RealNewsbody')
                result = fake;
                $('body').addClass('FakeNewsbody')
            } else {
                result = err;
            }
            document.querySelector('.result').innerHTML = result;
        } else {
            document.querySelector('.result').innerHTML = '<div class="spinner-grow text-primary" role="status"><span class="sr-only">Loading...</span></div><div class="spinner-grow text-secondary" role="status"><span class="sr-only">Loading...</span></div><div class="spinner-grow text-success" role="status"><span class="sr-only">Loading...</span></div><div class="spinner-grow text-danger" role="status"><span class="sr-only">Loading...</span></div><div class="spinner-grow text-warning" role="status"><span class="sr-only">Loading...</span></div><div class="spinner-grow text-info" role="status"><span class="sr-only">Loading...</span></div><div class="spinner-grow text-light" role="status"><span class="sr-only">Loading...</span></div><div class="spinner-grow text-dark" role="status"><span class="sr-only">Loading...</span></div>'
        }
    }
    Http.send();
});

