/* ========================================
   CHAT HISTORY
   ========================================
   The chat history sidebar is Chainlit's own (enabled by DATABASE_URL, see
   persistence/chat_history.py). It is styled in custom.css; no JavaScript needed.
   ======================================== */


/* ========================================
   LOGIN PAGE - RESPONSIVE LOGO POSITION
   ======================================== */

function positionLoginLogo() {
    if (window.location.pathname !== "/login") return;

    const form = document.querySelector('form:has(#email):has(#password)');
    const logo = document.querySelector('img.logo');

    if (!form || !logo) return;

    const container = form.parentElement;

    if (!container || container.contains(logo)) return;

    logo.classList.add("login-logo");
    container.insertBefore(logo, form);
}

const loginLogoObserver = new MutationObserver(() => {
    positionLoginLogo();
});

loginLogoObserver.observe(document.body, {
    childList: true,
    subtree: true
});

positionLoginLogo();


/* ========================================
   APPROVAL BUTTONS - IDENTIFICATION
   ======================================== */

function styleApprovalButtons() {
    document.querySelectorAll("button").forEach((button) => {
        const label = button.textContent.trim();

        if (label === "Approve") {
            button.setAttribute("data-helpdesk-action", "approve");
        } else if (label === "Reject") {
            button.setAttribute("data-helpdesk-action", "reject");
        }
    });
}

/* Detect dynamically rendered Chainlit buttons */
const approvalButtonsObserver = new MutationObserver(() => {
    styleApprovalButtons();
});

approvalButtonsObserver.observe(document.body, {
    childList: true,
    subtree: true
});

styleApprovalButtons();

/* ========================================
   NEW CHAT - NO CONFIRMATION DIALOG
   ========================================
   Chainlit asks "This will clear your current chat history" before a new chat.
   With chat history on (DATABASE_URL), the current chat is saved, not cleared,
   so the dialog only confuses users: start the new chat straight away.
   A fresh page load at "/" opens a new chat (new session, new thread). */

document.addEventListener(
    "click",
    (event) => {
        if (!event.target.closest("#new-chat-button")) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        window.location.assign("/");
    },
    true // capture phase: runs before Chainlit's own click handler opens the dialog
);
