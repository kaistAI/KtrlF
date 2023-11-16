export async function getCurrentTabId() {
    const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
    console.log("current focused: ", tab.id);
    return tab.id;
}

export const KTRLF_URL = "https://proved-quest-korean-involving.trycloudflare.com";
export const LINKPREVIEW_URL = "https://api.linkpreview.net/?key=a7db9ddfd68a1b848e6c02adfabd3f32&q=";