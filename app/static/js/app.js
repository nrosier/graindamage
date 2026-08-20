/*
 * The only hand-written JavaScript in the app: copy-to-clipboard for the rendered
 * commands. Delegated from the document so it keeps working on content HTMX swaps in
 * after load, and degrades to "select the text yourself" where the Clipboard API is
 * unavailable (any non-HTTPS origin that is not localhost).
 */
(function () {
  "use strict";

  document.addEventListener("click", function (event) {
    var trigger = event.target.closest("[data-copy]");
    if (!trigger) {
      return;
    }

    var source = document.querySelector(trigger.getAttribute("data-copy"));
    if (!source || !navigator.clipboard) {
      return;
    }

    var original = trigger.textContent;
    navigator.clipboard.writeText(source.textContent.trim()).then(
      function () {
        trigger.textContent = "Copied";
        trigger.classList.add("copied");
        window.setTimeout(function () {
          trigger.textContent = original;
          trigger.classList.remove("copied");
        }, 1600);
      },
      function () {
        trigger.textContent = "Press ⌘C";
        window.setTimeout(function () {
          trigger.textContent = original;
        }, 1600);
      }
    );
  });
})();
