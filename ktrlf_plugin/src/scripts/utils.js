export async function getCurrentTabId() {
    const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
    console.log("current focused: ", tab.id);
    return tab.id;
}

export const KTRLF_URL = "";
export const LINKPREVIEW_URL = "";