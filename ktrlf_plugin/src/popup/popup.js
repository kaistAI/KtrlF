async function getCurrentTabId() {
    const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
    return tab.id;
}

function clickIndexing() {
    (async () => {
        document.getElementById('spinner-overlay').style.setProperty('visibility', 'visible');

        const currentTabId = await getCurrentTabId();
        await chrome.tabs.sendMessage(currentTabId, {
            from: "popup",
            event: "clickIndexing"
        }, (res) => {
            if (res && res.flag) {
                const indexingSuccess = res.flag;
                console.log('popup indexing res: ', indexingSuccess);
                document.getElementById('spinner-overlay').style.setProperty('visibility', 'hidden');
            } else {
                console.log('popup indexing fail');
                document.getElementById('spinner-overlay').style.setProperty('visibility', 'hidden');
                document.getElementById('indexing-fail-overlay').style.setProperty('visibility', 'visible');
                if (res)
                    document.getElementById('indexing-fail-message').innerText = res.message;
            }
        });
    })();
}

function clickSearch() {
    (async () => {
        const inputValue = document.getElementById('input').value.trim();
        if (inputValue == "") return ;

        document.getElementById('search-button-icon').style.setProperty('display', 'none');
        document.getElementById('search-button-spinner').style.setProperty('display', 'block');

        const currentTabId = await getCurrentTabId();
        await chrome.tabs.sendMessage(currentTabId, {
            from: "popup",
            event: "clickSearch",
            query: inputValue,
            topk: parseInt(document.getElementById('range').max),
            defaultNumHighlight: parseInt(document.getElementById('range').value)
        }, (res) => {
            if (res){
                document.getElementById('search-button-icon').style.setProperty('display', 'block');
                document.getElementById('search-button-spinner').style.setProperty('display', 'none');    
            }
        });
    })();
}

function inputRange() {
    (async () => {
        const inputValue = parseInt(document.getElementById('range').value);

        const currentTabId = await getCurrentTabId();
        await chrome.tabs.sendMessage(currentTabId, {
            from: "popup",
            event: "inputRange",
            topk: inputValue
        });
    })();
}

function domReady() {
    return new Promise((resolve, reject) => {
        document.readyState === "interactive" || document.readyState === "complete" ? resolve() : document.addEventListener("DOMContentLoaded", resolve);
    }); 
}; 

domReady().then(() => {
    clickIndexing();
    document.getElementById('input').addEventListener("keypress", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            clickSearch();
        }
    });
    document.getElementById("search-button").addEventListener("click", (event) => {
        clickSearch();
    })
    document.getElementById('range').addEventListener("input", (event) => {
        inputRange();
    })

    chrome.runtime.connect({ name: "popup" });
})
