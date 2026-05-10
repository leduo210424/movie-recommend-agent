const userIdEl = document.getElementById('userId');
const topKEl = document.getElementById('topK');
const agentTypeEl = document.getElementById('agentType');
const queryEl = document.getElementById('query');
const statusEl = document.getElementById('status');
const decisionEl = document.getElementById('decisionCard');
const resultsEl = document.getElementById('results');
const explanationsEl = document.getElementById('explanations');
const recommendBtn = document.getElementById('recommendBtn');
const fillExampleBtn = document.getElementById('fillExampleBtn');

const EXAMPLE_QUERY = '想看轻松一点的电影';

function setStatus(text, tone = 'neutral') {
  statusEl.textContent = text;
  statusEl.style.color = tone === 'error' ? '#ffb4c2' : tone === 'success' ? '#8de8c8' : '#a7b7d6';
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function renderEmpty(target, text) {
  target.innerHTML = `<div class="empty-state">${escapeHtml(text)}</div>`;
}

function renderDecision(payload) {
  const parts = [];
  if (payload.agent_type) parts.push(`<div><strong>Agent:</strong> ${escapeHtml(payload.agent_type)}</div>`);
  if (payload.route) parts.push(`<div><strong>Route:</strong> ${escapeHtml(payload.route)}</div>`);
  if (payload.decision_reason) parts.push(`<div><strong>Decision:</strong> ${escapeHtml(payload.decision_reason)}</div>`);
  decisionEl.classList.remove('muted');
  decisionEl.innerHTML = parts.join('') || '暂无结果';
}

function renderResults(items) {
  if (!items || !items.length) {
    renderEmpty(resultsEl, '没有拿到推荐结果。');
    return;
  }

  resultsEl.innerHTML = items.map((item, index) => {
    const genres = (item.genres || []).map(escapeHtml).join(' / ');
    const reasons = (item.reasons || []).map((reason) => `<li>${escapeHtml(reason)}</li>`).join('');
    const components = item.components || {};
    return `
      <article class="card">
        <div class="card-title">
          <strong>${index + 1}. ${escapeHtml(item.title)}</strong>
          <span class="badge">${escapeHtml(String(item.release_year ?? 'N/A'))}</span>
        </div>
        <div class="meta">${genres}</div>
        <div class="meta">Score: ${Number(item.score ?? 0).toFixed(4)} · user ${Number(components.user_sim ?? 0).toFixed(3)} · rag ${Number(components.rag_sim ?? 0).toFixed(3)} · pop ${Number(components.popularity ?? 0).toFixed(3)}</div>
        <ul class="list">${reasons}</ul>
      </article>
    `;
  }).join('');
}

function renderExplanations(items) {
  if (!items || !items.length) {
    renderEmpty(explanationsEl, '没有拿到解释结果。');
    return;
  }

  explanationsEl.innerHTML = items.map((item, index) => {
    const features = (item.features || []).map((feature) => `<li>${escapeHtml(feature)}</li>`).join('');
    const evidence = (item.evidence || []).map((ev) => `<li>${escapeHtml(ev.title)} · ${escapeHtml(ev.reason)}</li>`).join('');
    const similarTitles = (item.similar_titles || []).map((title) => `<li>${escapeHtml(title)}</li>`).join('');
    const scores = item.scores || {};
    return `
      <article class="card">
        <div class="card-title">
          <strong>${index + 1}. ${escapeHtml(item.title)}</strong>
        </div>
        <div class="meta">user ${Number(scores.user_similarity ?? 0).toFixed(3)} · rag ${Number(scores.rag_similarity ?? 0).toFixed(3)} · pop ${Number(scores.popularity ?? 0).toFixed(3)}</div>
        <ul class="list"><li>核心理由</li>${features}</ul>
        ${evidence.length ? `<ul class="list"><li>历史证据</li>${evidence}</ul>` : ''}
        ${similarTitles.length ? `<ul class="list"><li>相似项</li>${similarTitles}</ul>` : ''}
        <div class="summary">${escapeHtml(item.summary || '')}</div>
      </article>
    `;
  }).join('');
}

// Generate a random session ID on load
const sessionId = Math.random().toString(36).substring(2, 15);

async function recommend() {
  const userId = userIdEl.value ? Number(userIdEl.value) : null;
  const topK = Number(topKEl.value || 3);
  const agentType = agentTypeEl.value || 'langgraph';
  const query = queryEl.value.trim();

  if (!query) {
    setStatus('请先输入一个 query。', 'error');
    return;
  }

  recommendBtn.disabled = true;
  setStatus('正在请求推荐结果...', 'neutral');

  try {
    const response = await fetch('/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, query, top_k: topK, agent: agentType, session_id: sessionId }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload?.detail || '请求失败');
    }

    renderDecision(payload);
    renderResults(payload.results);
    renderExplanations(payload.explanations);
    setStatus('推荐已生成。', 'success');
  } catch (error) {
    console.error(error);
    decisionEl.classList.add('muted');
    decisionEl.textContent = '暂无结果';
    renderEmpty(resultsEl, '请求失败，请稍后重试。');
    renderEmpty(explanationsEl, '请求失败，请稍后重试。');
    setStatus(`请求失败：${error.message}`, 'error');
  } finally {
    recommendBtn.disabled = false;
  }
}

fillExampleBtn.addEventListener('click', () => {
  userIdEl.value = 1;
  topKEl.value = 3;
  queryEl.value = EXAMPLE_QUERY;
  setStatus('已填充示例 query。', 'success');
});

recommendBtn.addEventListener('click', recommend);
queryEl.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
    recommend();
  }
});

renderEmpty(resultsEl, '等待一次推荐请求。');
renderEmpty(explanationsEl, '等待一次推荐请求。');
