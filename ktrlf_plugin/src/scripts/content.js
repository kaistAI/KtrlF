// import { LINKPREVIEW_URL } from "./utils.js";

timer = undefined;
function clearTimer(timer) {
    if (timer) {
        clearTimeout(timer);
    }
};

function showPreview(link) {
    function fetchPreview(url) {
        return fetch("https://api.linkpreview.net/?key=a7db9ddfd68a1b848e6c02adfabd3f32&q=" + encodeURIComponent(url))
        .then(function (response) {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error("Unable to fetch link preview data.");
            }
        });
    }

    // Create a preview box element
    var previewBox = document.getElementById("previewBox");
    previewBox.setAttribute('loadingState', "true");
  
    // Set the position of the preview box
    var linkRect = link.getBoundingClientRect();
    previewBox.style.top = document.body.scrollTop + linkRect.bottom + "px";
    previewBox.style.left = document.body.scrollLeft + linkRect.left + "px";
  
    // Set the content of the preview box
    previewBox.innerHTML = "Loading...";
  
    // Fetch the link preview data
    fetchPreview(link.href)
      .then(function (data) {
        if (previewBox.getAttribute('loadingState')!="true"){
            previewBox.style.setProperty("display", "none");
            return ;
        }

        console.log("received fetch preview",data)
        // Update the content of the preview box
        previewBox.innerHTML = `<div style="font-size: 13pt; font-weight: bold;">${data.title}</div><div style="font-size: 10pt">${data.description}</div>`
        // Show the preview box
        previewBox.style.setProperty("background", "yellow");
        previewBox.style.setProperty("display", "block");
        previewBox.style.setProperty("padding", "2px 4px");
        previewBox.style.setProperty("position", "fixed");
        previewBox.style.setProperty("z-index", "2147483647", 'important');
        previewBox.style.setProperty("color", 'black');
        previewBox.style.setProperty("border-radius", ".2em");
        previewBox.style.setProperty("box-shadow", "2px 2px 5px rgba(0,0,0,.4)");
      })
      .catch((err) => {console.log("cannot render preview")});
}
  
function hidePreview() {
    // Hide the preview box
    let previewBox = document.getElementById("previewBox");
    if (previewBox) {
        previewBox.style.display = "none";
        previewBox.setAttribute('loadingState', "false");
    }
}

function initPreview() {
    if (document.getElementById('previewBox')){
        hidePreview();
        console.log('preview is already existed');
        return ;
    }
    
    let previewBox = document.createElement("div");
    previewBox.id = 'previewBox';
    previewBox.style.setProperty("display", "none");
    document.getElementsByTagName("body")[0].appendChild(previewBox);
}

function replaceWordWithHTMLString(node, wordToReplace, htmlString) {
    if (node.nodeType === Node.TEXT_NODE) {
        // Create a new element with the HTML string
        const tempElement = document.createElement('div');
        tempElement.innerHTML = node.textContent.replace(wordToReplace, htmlString);

        // Replace the text node with the new element
        while (tempElement.firstChild) {
            node.parentNode.insertBefore(tempElement.firstChild, node);
        }
        node.parentNode.removeChild(node);

    } else if (node.nodeType === Node.ELEMENT_NODE) {
        // Recursively traverse child nodes for elements
        node.childNodes.forEach(childNode => {
            if (childNode.className !== 'previewAnchor')
                replaceWordWithHTMLString(childNode, wordToReplace, htmlString);
        });
    }
}

function clearHighlights() {
    let elementArr = document.querySelectorAll('[original-html]');
    elementArr.forEach((element) => {
        element.innerHTML = element.getAttribute('original-html');
        element.removeAttribute('original-html');
    })
}

function clearIndexingBox() {
    let elementArr = document.querySelectorAll('.indexing-box');
    elementArr.forEach((element) => {
        const originalBoxShadow = element.getAttribute('original-box-shadow');
        element.style.setProperty('box-shadow', originalBoxShadow);
        element.removeAttribute('original-box-shadow');
    })
}

function isElementInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
      rect.top >= 0 &&
      rect.left >= 0 &&
      rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
      rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
  }

function searchableElement(el) {
    MIN_TOKENS = 5;
    const validToken = (token) => {
      if (!token) return false;
  
      const alphaNum = token.match(/[a-zA-Z0-9]/g);
      return alphaNum && alphaNum.length > 0;
    };
  
    // Split by spaces, remove tokens without alphanumeric characters.
    const tokens = el.innerText.split(' ').filter(validToken);
    return tokens.length > MIN_TOKENS;
}

function updateHighlight(value) {
    let elementArr = document.querySelectorAll('.previewAnchor');
    for(element of elementArr) {
        const currentElementTopk = parseInt(element.getAttribute('topk'));
        const inputTopk = parseInt(value);

        // remove default anchor css
        element.style.setProperty('text-decoration', 'none');
        element.style.setProperty('color', 'inherit');

        // hide
        if (currentElementTopk > inputTopk){
            element.style.setProperty('pointer-events', 'none');
            element.getElementsByTagName('mark')[0].style.setProperty('background', 'none');
            element.getElementsByTagName('mark')[0].style.setProperty('color', 'unset');
        }
        // show
        else {
            element.style.setProperty('pointer-events', 'auto');
            element.getElementsByTagName('mark')[0].style.setProperty('background', 'mark');
            element.getElementsByTagName('mark')[0].style.setProperty('color', 'marktext');
        }
    }
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.from == 'background' && msg.event == 'appWakeUp') {
        initPreview();
    }
    if (msg.from == 'background' && msg.event == 'appClose') {
        clearHighlights();
        clearIndexingBox();
        hidePreview();
    }

    JOIN_TOKEN = '\n';
    if (msg.from == "popup" && msg.event == "clickIndexing") {
        clearHighlights();
        clearIndexingBox();
        initPreview();
        
        let contextElementArray = Array.from(document.querySelectorAll("p,ol,ul"))
        //   .filter(isElementInViewport) 
          .filter(searchableElement)
          .filter((el) => el.offsetParent !== null);
        if (contextElementArray.length == 0){
            sendResponse({'flag': false, 'message': "There are no tags available for indexing."});
            return ;
        }

        let contextElementText = contextElementArray.map((el)=>{return el.innerText.trim();}).join(JOIN_TOKEN);

        chrome.runtime.sendMessage({
            from: "content",
            event: "requestIndexing",
            context: contextElementText
        }, (res) => {
            if (res && res.flag){
                console.log("content Indexing finish");
                contextElementArray.forEach((contextElement) => {
                    contextElement.className += " indexing-box";
                    const originalBoxShadow = contextElement.style.getPropertyValue('box-shadow');
                    contextElement.setAttribute('original-box-shadow', originalBoxShadow);
                    contextElement.style.setProperty('box-shadow', 'rgba(3, 102, 214, 0.3) 0px 0px 0px 3px');
                })
                sendResponse({'flag': true, 'message': "Indexing Sucess."});
            } else {
                console.log("content Indexing fail: ", res);
                sendResponse({'flag': false, 'message': "Indexing server is busy. Please retry after page reloading."});
            }
        })
        return true;
    }

    if (msg.from == "popup" && msg.event == "clickSearch") {
        clearHighlights();
        hidePreview();

        // request to model
        let contextElementArray = Array.from(document.querySelectorAll("p,ol,ul"))
        //   .filter(isElementInViewport)
          .filter(searchableElement)
          .filter((el) => el.offsetParent !== null);
        let contextElementText = contextElementArray.map((el)=>{return el.innerText.trim();}).join(JOIN_TOKEN);

        let offset = -JOIN_TOKEN.length;
        const contextElementInfosWithAccumIndexArray = contextElementArray.map((contextElement) => {
            const startIndex = offset + JOIN_TOKEN.length;
            const endIndex = startIndex + contextElement.innerText.trim().length;
            offset = endIndex - 1 + JOIN_TOKEN.length;
            return {'start': startIndex, 'end': endIndex, 'element': contextElement, 'originalHTML': contextElement.innerHTML};
        });
        console.log("searching target context: ", contextElementInfosWithAccumIndexArray);
        const findContextElementInfoByJoinedIndex = (start, end) => {
            for(tup of contextElementInfosWithAccumIndexArray){
                if (tup.start<=start && end<=tup.end) {
                    return tup
                }
            }
            return null;
        }

        chrome.runtime.sendMessage({
            from: "content",
            event: "requestSearch",
            context: contextElementText,
            query: msg.query,
            topk: msg.topk
        }, (preds) => {
            console.log('content search preds: ', preds);
            for(i=0; i<preds.length; i++){
                let pred = preds[i];
                let contextElementInfo = findContextElementInfoByJoinedIndex(pred.start, pred.end);
                let contextElement = contextElementInfo?contextElementInfo.element:null;
                const originalHTML = contextElementInfo?contextElementInfo.originalHTML:null;
                if (pred.text!="" && pred.start>=0 && pred.end>=0 && contextElement){
                    replaceWordWithHTMLString(contextElement, pred.text, `<a href="${pred.wikipedia_link}" class="previewAnchor" topk=${i+1}><mark>${pred.text}</mark></a>`);
                    contextElement.setAttribute('original-html', originalHTML);
                }
            };

            // add event handler for each preview
            console.log('content preview anchors');
            let anchors = document.querySelectorAll('.previewAnchor');
            anchors.forEach(a => {
                a.addEventListener('mouseover', () => {
                    if (timer) clearTimer(timer);
		            timer = setTimeout( () => {timer=undefined; showPreview(a)}, 800);
                })
                a.addEventListener('mouseout', () => {
                    clearTimer(timer);
                    hidePreview();
                })
            });

            // show only default number of preds
            updateHighlight(msg.defaultNumHighlight);
            sendResponse({'flag': true, 'message': "Search Sucess."});
        });
        return true;
    }
    
    if (msg.from == "popup" && msg.event == "inputRange") {
        updateHighlight(msg.topk);
    }
});

