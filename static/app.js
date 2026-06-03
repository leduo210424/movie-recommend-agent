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
const newSessionBtn = document.getElementById('newSessionBtn');
const sessionDisplay = document.getElementById('sessionDisplay');
const conversationEl = document.getElementById('conversation');
const turnCountEl = document.getElementById('turnCount');
const streamModeEl = document.getElementById('streamMode');

const EXAMPLE_QUERY = '想看轻松一点的电影';

// 全局状态
let sessionId = Math.random().toString(36).substring(2, 15);
let currentResults = []; // 当前推荐结果，用于反馈
let turnNumber = 0;

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
    const movieId = item.movie_id;
    return `
      <article class="card" data-movie-id="${movieId}">
        <div class="card-title">
          <strong>${index + 1}. ${escapeHtml(item.title)}</strong>
          <span class="badge">${escapeHtml(String(item.release_year ?? 'N/A'))}</span>
        </div>
        <div class="meta">${genres}</div>
        <div class="meta">Score: ${Number(item.score ?? 0).toFixed(4)} · user ${Number(components.user_sim ?? 0).toFixed(3)} · rag ${Number(components.rag_sim ?? 0).toFixed(3)} · pop ${Number(components.popularity ?? 0).toFixed(3)}</div>
        <ul class="list">${reasons}</ul>
        <div class="feedback-actions">
          <button class="fb-btn fb-like" data-action="like" data-movie-id="${movieId}" data-title="${escapeHtml(item.title)}" data-genres="${escapeHtml((item.genres || []).join(','))}" title="喜欢">👍 喜欢</button>
          <button class="fb-btn fb-dislike" data-action="dislike" data-movie-id="${movieId}" data-title="${escapeHtml(item.title)}" data-genres="${escapeHtml((item.genres || []).join(','))}" title="不喜欢">👎 不喜欢</button>
        </div>
      </article>
    `;
  }).join('');

  // 绑定反馈事件
  resultsEl.querySelectorAll('.fb-btn').forEach(btn => {
    btn.addEventListener('click', handleFeedback);
  });
}

function renderExplanations(items) {
  console.log('[renderExplanations] called with', items);
  if (!items || !items.length) {
    renderEmpty(explanationsEl, '没有拿到解释结果。');
    return;
  }

  console.log('[renderExplanations] rendering', items.length, 'explanations');
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

// ── 对话历史渲染 ──

function addConversationTurn(query, results) {
  turnNumber++;
  turnCountEl.textContent = `(第 ${turnNumber} 轮)`;

  // 移除空状态
  const emptyState = conversationEl.querySelector('.empty-state');
  if (emptyState) emptyState.remove();

  const movieTitles = results.map(r => escapeHtml(r.title)).join('、');

  const turnHtml = `
    <div class="conv-turn">
      <div class="conv-query">
        <span class="conv-role">🫵 用户</span>
        <span class="conv-text">${escapeHtml(query)}</span>
      </div>
      <div class="conv-response">
        <span class="conv-role">🤖 助手</span>
        <span class="conv-text">推荐了: ${movieTitles || '无结果'}</span>
      </div>
    </div>
  `;
  conversationEl.insertAdjacentHTML('beforeend', turnHtml);
  conversationEl.scrollTop = conversationEl.scrollHeight;
}

// ── 反馈处理 ──

async function handleFeedback(event) {
  const btn = event.currentTarget;
  const movieId = Number(btn.dataset.movieId);
  const title = btn.dataset.title;
  const genres = btn.dataset.genres ? btn.dataset.genres.split(',') : [];
  const action = btn.dataset.action;
  const userId = userIdEl.value ? Number(userIdEl.value) : null;

  if (!userId) {
    setStatus('请先输入 User ID 再提交反馈。', 'error');
    return;
  }

  // 禁用所有反馈按钮，防止重复点击
  const card = btn.closest('.card');
  if (!card) return;
  const allBtns = card.querySelectorAll('.fb-btn');
  allBtns.forEach(b => b.disabled = true);

  // 高亮选中的按钮
  btn.classList.add('fb-selected');

  setStatus(`正在提交反馈: ${action === 'like' ? '喜欢' : '不喜欢'} "${title}"...`, 'neutral');

  try {
    const response = await fetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        movie_id: movieId,
        feedback: action,
        movie_title: title,
        movie_genres: genres,
        session_id: sessionId,
      }),
    });

    const payload = await response.json();
    if (!response.ok) throw new Error(payload?.detail || '反馈请求失败');

    setStatus(`反馈已记录: ${action === 'like' ? '👍 喜欢' : '👎 不喜欢'} "${title}"。下次推荐会据此调整。`, 'success');
  } catch (error) {
    console.error(error);
    setStatus(`反馈提交失败: ${error.message}`, 'error');
    // 恢复按钮
    allBtns.forEach(b => b.disabled = false);
    btn.classList.remove('fb-selected');
  }
}

// ── 推荐请求 ──

let activeWs = null;  // 当前活跃的 WebSocket 连接

async function recommend() {
  if (streamModeEl.checked) {
    return recommendStream();
  }
  return recommendRest();
}

async function recommendRest() {
  const userId = userIdEl.value ? Number(userIdEl.value) : null;
  const topK = Number(topKEl.value || 3);
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
      body: JSON.stringify({ user_id: userId, query, top_k: topK, session_id: sessionId }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload?.detail || '请求失败');
    }

    currentResults = payload.results || [];

    renderDecision(payload);
    renderResults(payload.results);
    renderExplanations(payload.explanations);
    addConversationTurn(query, currentResults);
    setStatus('推荐已生成。可以通过 👍👎 按钮给出反馈，下次推荐会据此调整。', 'success');
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

function recommendStream() {
  const userId = userIdEl.value ? Number(userIdEl.value) : null;
  const topK = Number(topKEl.value || 3);
  const query = queryEl.value.trim();

  if (!query) {
    setStatus('请先输入一个 query。', 'error');
    return;
  }

  // 关闭之前的连接
  if (activeWs) {
    activeWs.close();
    activeWs = null;
  }

  recommendBtn.disabled = true;
  decisionEl.classList.remove('muted');
  decisionEl.innerHTML = '<div id="streamStatus" class="stream-live">Agent 推理中...</div>';
  renderEmpty(resultsEl, '等待流式结果...');
  renderEmpty(explanationsEl, '等待流式结果...');

  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${location.host}/ws/recommend`;
  const ws = new WebSocket(wsUrl);
  activeWs = ws;
  let streamResults = [];
  let streamCancelFn = null;

  ws.onopen = () => {
    setStatus('WebSocket 已连接，正在流式推理...', 'neutral');
    ws.send(JSON.stringify({
      type: 'recommend',
      user_id: userId,
      query: query,
      top_k: topK,
      session_id: sessionId,
    }));
    // 开启"取消"按钮
    streamCancelFn = () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'cancel' }));
        setStatus('已发送取消信号。', 'neutral');
      }
    };
    recommendBtn.textContent = '取消';
    recommendBtn.onclick = streamCancelFn;
  };

  ws.onmessage = (event) => {
    const evt = JSON.parse(event.data);
    switch (evt.event) {
      case 'thinking':
        document.getElementById('streamStatus').textContent =
          `🔄 ${evt.data}`;
        break;
      case 'token':
        document.getElementById('streamStatus').textContent =
          `💭 ${(evt.data || '').substring(0, 80)}...`;
        break;
      case 'tool_call':
        document.getElementById('streamStatus').textContent =
          `🔧 调用工具: ${evt.data.tool}`;
        break;
      case 'observation':
        document.getElementById('streamStatus').textContent =
          `📊 ${evt.data.tool} 完成，找到 ${evt.data.movie_count} 部候选`;
        break;
      case 'reasoning_done':
        document.getElementById('streamStatus').textContent =
          `✅ Agent 推理完成`;
        break;
      case 'results':
        streamResults = evt.data.movies || [];
        currentResults = streamResults;
        console.log('[results] movies:', streamResults.length, 'explanations:', (evt.data.explanations || []).length);
        // 构建兼容 renderDecision 的 payload
        renderDecision({
          route: evt.data.route,
          decision_reason: `Agent 经过 ${evt.data.iterations} 轮推理完成 (session: ${evt.data.session_id})`,
        });
        renderResults(streamResults);
        renderExplanations(evt.data.explanations || []);
        addConversationTurn(query, streamResults);
        setStatus('流式推荐已完成。', 'success');
        break;
      case 'cancelled':
        setStatus('推荐已取消。', 'neutral');
        break;
      case 'error':
        setStatus(`流式错误: ${evt.data}`, 'error');
        break;
      case 'done':
        var streamStatusEl = document.getElementById('streamStatus');
        if (streamStatusEl) {
            streamStatusEl.textContent = `✅ 完成，共推荐 ${streamResults.length} 部电影`;
        }
        cleanupStream();
        break;
    }
  };

  ws.onerror = (err) => {
    console.error('WebSocket error:', err);
    setStatus('WebSocket 连接错误。', 'error');
    cleanupStream();
  };

  ws.onclose = () => {
    cleanupStream();
  };

  function cleanupStream() {
    recommendBtn.textContent = '生成推荐';
    recommendBtn.onclick = recommend;
    recommendBtn.disabled = false;
    if (activeWs === ws) activeWs = null;
  }
}

// ── Session 管理 ──

function updateSessionDisplay() {
  sessionDisplay.textContent = sessionId;
}

async function resetSession() {
  try {
    await fetch(`/session/reset?session_id=${encodeURIComponent(sessionId)}`, { method: 'POST' });
  } catch (e) {
    // 即使后端请求失败也继续重置前端
  }

  sessionId = Math.random().toString(36).substring(2, 15);
  turnNumber = 0;
  currentResults = [];
  updateSessionDisplay();
  turnCountEl.textContent = '';
  conversationEl.innerHTML = '<div class="empty-state">新对话已开始。输入查询开始对话。</div>';
  decisionEl.classList.add('muted');
  decisionEl.textContent = '暂无结果';
  renderEmpty(resultsEl, '等待新查询。');
  renderEmpty(explanationsEl, '等待新查询。');
  setStatus('已开始新对话。', 'success');
}

updateSessionDisplay();

// ── 事件绑定 ──

fillExampleBtn.addEventListener('click', () => {
  userIdEl.value = 1;
  topKEl.value = 3;
  queryEl.value = EXAMPLE_QUERY;
  setStatus('已填充示例 query。', 'success');
});

newSessionBtn.addEventListener('click', resetSession);
recommendBtn.addEventListener('click', recommend);
queryEl.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
    recommend();
  }
});

// 初始状态
renderEmpty(resultsEl, '等待一次推荐请求。');
renderEmpty(explanationsEl, '等待一次推荐请求。');
