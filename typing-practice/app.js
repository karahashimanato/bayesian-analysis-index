(function () {
  "use strict";

  const CHAPTERS = window.SNIPPETS_DATA.chapters;

  const els = {
    chapterList: document.getElementById("chapter-list"),
    viewChapters: document.getElementById("view-chapters"),
    viewGame: document.getElementById("view-game"),
    gameTitle: document.getElementById("game-title"),
    statTime: document.getElementById("stat-time"),
    statAccuracy: document.getElementById("stat-accuracy"),
    statScore: document.getElementById("stat-score"),
    statProgress: document.getElementById("stat-progress"),
    panelStart: document.getElementById("panel-start"),
    panelPlay: document.getElementById("panel-play"),
    panelResult: document.getElementById("panel-result"),
    startSummary: document.getElementById("start-summary"),
    btnStart: document.getElementById("btn-start"),
    btnBack: document.getElementById("btn-back"),
    btnRetry: document.getElementById("btn-retry"),
    btnToList: document.getElementById("btn-to-list"),
    targetDisplay: document.getElementById("target-display"),
    typingInput: document.getElementById("typing-input"),
    resultTime: document.getElementById("result-time"),
    resultAccuracy: document.getElementById("result-accuracy"),
    resultScore: document.getElementById("result-score"),
  };

  /** @type {{chapter: object, order: object[], index: number, prevValue: string,
   *          totalKeystrokes: number, correctKeystrokes: number,
   *          startedAt: number, timerId: number|null}} */
  let session = null;

  function shuffle(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  function renderChapterList() {
    els.chapterList.innerHTML = "";
    for (const chapter of CHAPTERS) {
      const card = document.createElement("button");
      card.className = "chapter-card";
      card.innerHTML = `
        <h3>${escapeHtml(chapter.title)}</h3>
        <div class="meta">${chapter.problems.length}問 &middot;
          <a href="../${chapter.source}" target="_blank" rel="noopener">${chapter.source}</a>
        </div>`;
      card.addEventListener("click", () => openChapter(chapter));
      els.chapterList.appendChild(card);
    }
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function openChapter(chapter) {
    session = {
      chapter,
      order: shuffle(chapter.problems),
      index: 0,
      prevValue: "",
      totalKeystrokes: 0,
      correctKeystrokes: 0,
      startedAt: 0,
      timerId: null,
    };
    els.gameTitle.textContent = chapter.title;
    els.startSummary.textContent = `全 ${chapter.problems.length} 問。正確に入力すると自動で次の問題に進みます。`;
    showPanel("start");
    updateStats(0);
    els.viewChapters.classList.add("hidden");
    els.viewGame.classList.remove("hidden");
  }

  function backToList() {
    stopTimer();
    session = null;
    els.viewGame.classList.add("hidden");
    els.viewChapters.classList.remove("hidden");
  }

  function showPanel(name) {
    els.panelStart.classList.toggle("hidden", name !== "start");
    els.panelPlay.classList.toggle("hidden", name !== "play");
    els.panelResult.classList.toggle("hidden", name !== "result");
  }

  function startChapter() {
    session.startedAt = performance.now();
    session.timerId = setInterval(tick, 200);
    showPanel("play");
    loadProblem();
  }

  function tick() {
    const elapsedSec = (performance.now() - session.startedAt) / 1000;
    updateStats(elapsedSec);
  }

  function updateStats(elapsedSec) {
    const mm = String(Math.floor(elapsedSec / 60)).padStart(2, "0");
    const ss = String(Math.floor(elapsedSec % 60)).padStart(2, "0");
    els.statTime.textContent = `${mm}:${ss}`;

    const total = session ? session.totalKeystrokes : 0;
    const correct = session ? session.correctKeystrokes : 0;
    const accuracy = total > 0 ? (correct / total) * 100 : 0;
    els.statAccuracy.textContent = `${accuracy.toFixed(2)}%`;

    const score = elapsedSec > 0 ? (correct / elapsedSec) * 60 : 0;
    els.statScore.textContent = score.toFixed(1);

    const idx = session ? Math.min(session.index + 1, session.order.length) : 0;
    const totalProblems = session ? session.order.length : 0;
    els.statProgress.textContent = `${idx} / ${totalProblems}`;
  }

  function stopTimer() {
    if (session && session.timerId !== null) {
      clearInterval(session.timerId);
      session.timerId = null;
    }
  }

  function loadProblem() {
    const problem = session.order[session.index];
    session.prevValue = "";
    els.typingInput.value = "";
    renderTarget(problem.text, "");
    els.typingInput.disabled = false;
    els.typingInput.focus();
  }

  function renderTarget(target, typed) {
    let html = "";
    for (let i = 0; i < target.length; i++) {
      const ch = target[i];
      const display = ch === "\n" ? "↵\n" : escapeHtml(ch);
      let cls = "char-pending";
      if (i < typed.length) {
        cls = typed[i] === ch ? "char-correct" : "char-wrong";
      } else if (i === typed.length) {
        cls += " char-current";
      }
      html += `<span class="${cls}">${display}</span>`;
    }
    els.targetDisplay.innerHTML = html;
  }

  function onTypingInput() {
    const problem = session.order[session.index];
    const target = problem.text;
    const newValue = els.typingInput.value;
    const prevValue = session.prevValue;

    if (newValue.length === prevValue.length + 1 && newValue.startsWith(prevValue)) {
      const idx = prevValue.length;
      session.totalKeystrokes++;
      if (newValue[idx] === target[idx]) session.correctKeystrokes++;
    }
    session.prevValue = newValue;

    renderTarget(target, newValue);
    updateStats((performance.now() - session.startedAt) / 1000);

    if (newValue === target) {
      advanceProblem();
    }
  }

  function advanceProblem() {
    els.typingInput.disabled = true;
    setTimeout(() => {
      session.index++;
      if (session.index >= session.order.length) {
        finishChapter();
      } else {
        loadProblem();
      }
    }, 250);
  }

  function finishChapter() {
    stopTimer();
    const elapsedSec = (performance.now() - session.startedAt) / 1000;
    updateStats(elapsedSec);
    els.resultTime.textContent = els.statTime.textContent;
    els.resultAccuracy.textContent = els.statAccuracy.textContent;
    els.resultScore.textContent = els.statScore.textContent;
    showPanel("result");
  }

  els.typingInput.addEventListener("input", onTypingInput);
  els.typingInput.addEventListener("paste", (e) => e.preventDefault());
  els.typingInput.addEventListener("keydown", (e) => {
    if (e.key === "Tab") { e.preventDefault(); document.execCommand("insertText", false, "    "); }
  });

  els.btnStart.addEventListener("click", startChapter);
  els.btnBack.addEventListener("click", backToList);
  els.btnToList.addEventListener("click", backToList);
  els.btnRetry.addEventListener("click", () => openChapter(session.chapter));

  renderChapterList();
})();
