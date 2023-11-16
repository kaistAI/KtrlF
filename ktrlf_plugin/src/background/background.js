import { getCurrentTabId, KTRLF_URL } from "../scripts/utils.js";

async function indexing(context){
    let res = await fetch(KTRLF_URL, {
        
        headers : {
            'Content-Type' : 'application/json'
        },
        method : 'POST',
        body : JSON.stringify({
            'type':'indexing',
            'context': context
        })
    })
    if (res.ok) return res.json();
    else throw Error('Indexing Server Error');
};

async function getPreds(context, query, topk){
    let res = await fetch(KTRLF_URL, {
        headers : {
            'Content-Type' : 'application/json'
        },
        method : 'POST',
        body : JSON.stringify({
            'type':'search',
            'query': query,
            'context': context,
        })
    })
    if (res.ok){
        let resJson = await res.json();
        return resJson.result;
    }
    else throw Error('Something went wrong');
};

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    console.log("background input: ", msg);
    if (msg.event == "requestIndexing") {
        indexing(msg.context).then((res) => {
            console.log("background indexing res: ", res);
            sendResponse(res);
        }).catch((err) => {
            console.log("background indexing fail: ", err);
            sendResponse(null);
        })
        // Important! Return true to indicate you want to send a response asynchronously
        return true;
    }
    if (msg.event == "requestSearch") {
        getPreds(msg.context, msg.query, msg.topk).then((preds) => {
            console.log("background search res (before filtered): ", preds);
            let filtered_preds=preds.filter((pred) => {return pred.score>=2.0});
            console.log("background search res (after filtered): ", filtered_preds);
            sendResponse(filtered_preds);
        })
        return true;
    }
});

chrome.runtime.onConnect.addListener((port) => {
    if (port.name === "popup") {
        getCurrentTabId().then((openedTabId)=>{
            chrome.tabs.sendMessage(openedTabId, {
                from: "background",
                event: "appWakeUp"
            })
            
            port.onDisconnect.addListener(() => {
                console.log("popup has been closed");
                chrome.tabs.sendMessage(openedTabId, {
                    from: "background",
                    event: "appClose"
                });
             });
        })
    }
});