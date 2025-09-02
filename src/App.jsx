import "./App.css"

import React, { useMemo, useState, useEffect, useRef } from "react";

/**
 * PICSY-Note Prototype (Single-file React App)
 * ------------------------------------------------------
 * v3 — Rich social features + random like simulator + who-liked-whom visibility
 *
 * 新要件:
 * - 投稿のリッチ化: 画像/タグ/改行、投稿作成UI、検索(テキスト/タグ)、並び替え(新着/ホット/作者のc)
 * - 誰が誰にLikeしたか: ポスト別履歴、グローバル台帳、Likeフロー行列(Σδ)
 * - 自動価値移転: ランダムLikeシミュレータ(他ユーザーが自動で評価)
 * - プロフィール: 任意ユーザーの c / 予算 / PP / 投稿数 / 受領δ / 送信δ を可視化
 * - 既存のPICSYコア(仮想中央銀行法/自然回収/メンバー追加)は維持
 *
 * 数学:
 * - E' = E - B + (B D)/(N-1)（左固有ベクトル c, sum(c)=N）
 * - Like: δ = α·c_b ⇒ α = δ / c_b
 * - Recovery: offdiag *= (1-γ), diag ← diag + γ(1-diag)
 * - AddMember: x=1/N (既存 c/予算 不変), 新規 c=1, 予算0
 */

// ---------- Utilities ----------
const fmt = (x, digits = 3) => (Number.isNaN(x) ? "NaN" : Number(x).toFixed(digits));
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const rnd = (lo, hi) => lo + Math.random() * (hi - lo);
const deepCopy = (E) => E.map((row) => row.slice());
const sum = (arr) => arr.reduce((a, b) => a + b, 0);
const zeros = (n, m) => Array.from({ length: n }, () => Array(m).fill(0));
const nowMs = () => Date.now();

/**
 * 
 * @param {Array<Array<number>>} E 
 * @returns {Array<Array<number>>} Matrix
 */
const ensureRowStochastic = (E) => {
  const n = E.length;
  for (let i = 0; i < n; i++) {
    let rowSum = sum(E[i]);
    if (rowSum === 0) {
      for (let j = 0; j < n; j++) E[i][j] = 1 / n;
      rowSum = 1;
    }
    if (Math.abs(rowSum - 1) > 1e-10) {
      for (let j = 0; j < n; j++) E[i][j] /= rowSum;
    }
  }
  return E;
};

// ---------- PICSY Math ----------
/**
 * 
 * @param {number[]} v 
 * @param {number[][]} E 
 * @returns {number[]}
 */
function leftMultiplyEPrime(v, E) {
  const n = E.length;
  if (n <= 1) return v.slice();
  const out = new Array(n).fill(0);
  const vE = new Array(n).fill(0);
  const vB = new Array(n).fill(0);
  let S = 0;
  for (let j = 0; j < n; j++) {
    vB[j] = v[j] * E[j][j];
    S += vB[j];
  }
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += v[i] * E[i][j];
    vE[j] = s;
  }
  const inv = 1 / (n - 1);
  for (let j = 0; j < n; j++) out[j] = vE[j] - vB[j] + inv * (S - vB[j]);
  return out;
}

/**
 * 
 * @param {number[]} v 
 * @param {number} targetSum 
 * @returns {number}
 */
function normalizeToSum(v, targetSum) {
  const s = sum(v);
  if (s === 0) return v.map(() => targetSum / v.length);
  const scale = targetSum / s;
  return v.map((x) => x * scale);
}

/**
 * 二つのベクトルの同じインデックスの要素の差を合計する
 * @param {number[]} a 
 * @param {number[]} b 
 * @returns {number}
 */
function l1Diff(a, b) {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += Math.abs(a[i] - b[i]);
  return d;
}

/**
 * 
 * @param {number[][]} E 
 * @param {{maxIter?:number, tol?:number, warmStart:number[]}} opts 
 * @returns 
 */
function powerIterationLeft(E, opts = {}) {
  const n = E.length;
  const { maxIter = 1000, tol = 1e-10, warmStart } = opts;
  let v = warmStart && warmStart.length === n ? warmStart.slice() : Array(n).fill(1 / n);
  v = normalizeToSum(v, 1);
  for (let k = 0; k < maxIter; k++) {
    const vNext = leftMultiplyEPrime(v, E);
    const vNextNorm = normalizeToSum(vNext, 1);
    if (l1Diff(vNextNorm, v) < tol) {
      v = vNextNorm;
      break;
    }
    v = vNextNorm;
  }
  return v.map((x) => x * n); // sum = n
}

function applyNaturalRecovery(E, gamma) {
  const n = E.length;
  const out = deepCopy(E);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j) out[i][j] *= 1 - gamma;
    }
    out[i][i] = out[i][i] + gamma * (1 - out[i][i]);
  }
  return ensureRowStochastic(out);
}

function applyLike(E, c, b, s, delta) {
  if (b === s) throw new Error("Self-like is not allowed.");
  const alpha = delta / c[b];
  if (alpha < 0) throw new Error("Negative alpha.");
  if (alpha > E[b][b] + 1e-12) throw new Error("Insufficient budget.");
  const out = deepCopy(E);
  out[b][b] -= alpha;
  out[b][s] += alpha;
  return ensureRowStochastic(out);
}

function addMember(E, c, name) {
  const N = E.length;
  const x = 1 / N;
  const K = N + 1;
  const Enew = zeros(K, K);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) Enew[i][j] = i === j ? E[i][j] : (1 - x) * E[i][j];
    Enew[i][N] = x * (1 - E[i][i]);
  }
  for (let j = 0; j < N; j++) Enew[N][j] = c[j] / N;
  Enew[N][N] = 0;
  ensureRowStochastic(Enew);
  const cnew = powerIterationLeft(Enew, { warmStart: normalizeToSum([...c.map((x) => x / N), 1 / N], 1) });
  return { E: Enew, c: cnew, addedName: name };
}

// ---------- Self Tests (console) ----------
function runSelfTests() {
  let passed = 0;
  let failed = 0;
  const assert = (cond, msg) => {
    if (!cond) {
      console.error("TEST FAIL:", msg);
      failed++;
    } else {
      console.log("TEST OK:", msg);
      passed++;
    }
  };
  try {
    // Base matrix (3x3): diag=0.2, off=0.4
    const E0 = ensureRowStochastic([
      [0, 0.65, 0.35],
      [0.50, 0, 0.50],
      [0.25, 0.75, 0],
    ]);
    assert(E0.every((r) => Math.abs(sum(r) - 1) < 1e-12), "row sums to 1");

    const c0 = powerIterationLeft(E0);
    console.log("c0", c0);
    
    assert(Math.abs(sum(c0) - 3) < 1e-8, "sum(c)=N");

    // applyLike correctness
    const E1 = applyLike(E0, c0, 0, 1, 0.05);
    const alpha = 0.05 / c0[0];
    assert(Math.abs(E1[0][0] - (E0[0][0] - alpha)) < 1e-10, "applyLike reduces buyer diag by alpha");
    assert(Math.abs(E1[0][1] - (E0[0][1] + alpha)) < 1e-10, "applyLike increases buyer->seller by alpha");

    // recovery monotonicity
    const E2 = applyNaturalRecovery(E0, 0.1);
    assert(E2[0][0] > E0[0][0], "recovery increases diag");
    assert(E2[0][1] < E0[0][1], "recovery decreases off-diag");

    // addMember invariants
    const { E: E3, c: c3 } = addMember(E0, c0, "new");
    assert(Math.abs(E3[0][0] - E0[0][0]) < 1e-12, "addMember keeps budgets");
    assert(Math.abs(sum(c3) - E3.length) < 1e-6, "sum(c)=N after addMember");
    assert(Math.abs(c3[c3.length - 1] - 1) < 1e-3, "new member c=1");
  } catch (e) {
    console.warn("Self-tests exception:", e);
  }
  console.log(`PICSY self-tests: passed=${passed}, failed=${failed}`);
}

// ---------- UI Components ----------
function Section({ title, children, actions, footer }) {
  return (
    <div className="mb-6 rounded-2xl border p-4 shadow-sm bg-white">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-lg font-semibold">{title}</h2>
        <div className="flex gap-2">{actions}</div>
      </div>
      {children}
      {footer && <div className="mt-3 text-xs text-gray-500">{footer}</div>}
    </div>
  );
}

function MatrixTable({ title, matrix, highlightDiag = false, c, pp }) {
  if (!matrix || matrix.length === 0) return null;
  const n = matrix.length;
  const showStats = Array.isArray(c) && Array.isArray(pp) && c.length === n && pp.length === n;
  return (
    <div className="overflow-auto">
      <div className="text-sm font-medium mb-2">{title}</div>
      <table className="min-w-max text-right text-sm">
        <thead>
          <tr>
            <th className="px-2 py-1 text-left">row/col</th>
            {showStats && (
              <>
                <th className="px-2 py-1">貢献度c</th>
                <th className="px-2 py-1">購買力PP</th>
              </>
            )}
            {Array.from({ length: n }, (_, j) => (
              <th key={j} className="px-2 py-1">{j + 1}</th>
            ))}
            <th className="px-2 py-1">row Σ</th>
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i} className="border-t">
              <td className="px-2 py-1 text-left font-medium">{i + 1}</td>
              {showStats && (
                <>
                  <td className="px-2 py-1">{fmt(c[i])}</td>
                  <td className="px-2 py-1">{fmt(pp[i])}</td>
                </>
              )}
              {row.map((x, j) => (
                <td
                  key={j}
                  className={
                    "px-2 py-1 " +
                    (highlightDiag && i === j ? "bg-amber-50 font-semibold" : "")
                  }
                  title={`E[${i + 1},${j + 1}]`}
                >
                  {fmt(x)}
                </td>
              ))}
              <td className="px-2 py-1 font-semibold">{fmt(sum(row))}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}


function VectorTable({ title, vec, scale = 1, totalLabel = "Σ" }) {
  if (!vec || vec.length === 0) return null;
  const total = sum(vec) * scale;
  return (
    <div className="overflow-auto">
      <div className="text-sm font-medium mb-2">{title}</div>
      <table className="min-w-max text-right text-sm">
        <thead>
          <tr>
            {vec.map((_, j) => (
              <th key={j} className="px-2 py-1">
                {j + 1}
              </th>
            ))}
            <th className="px-2 py-1">{totalLabel}</th>
          </tr>
        </thead>
        <tbody>
          <tr className="border-t">
            {vec.map((x, j) => (
              <td key={j} className="px-2 py-1">
                {fmt(x * scale)}
              </td>
            ))}
            <td className="px-2 py-1 font-semibold">{fmt(total)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function Avatar({ name }) {
  const hue = Math.abs(hashCode(name)) % 360;
  const bg = `hsl(${hue}, 70%, 90%)`;
  const fg = `hsl(${hue}, 70%, 25%)`;
  const initials = name.slice(0, 2).toUpperCase();
  return (
    <div
      className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold"
      style={{ background: bg, color: fg }}
    >
      {initials}
    </div>
  );
}

function hashCode(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) h = ((h << 5) - h + str.charCodeAt(i)) | 0;
  return h;
}

function CommentList({ post, users, onAdd }) {
  const [text, setText] = useState("");
  return (
    <div className="mt-3">
      <div className="text-xs text-gray-500 mb-1">Comments ({post.comments.length})</div>
      <ul className="text-sm space-y-1 mb-2">
        {post.comments.slice(-5).map((c) => (
          <li key={c.id} className="flex justify-between">
            <span className="flex items-center gap-2">
              <Avatar name={users[c.author].handle} />@{users[c.author].handle}: {c.text}
            </span>
            <span className="text-xs text-gray-500">{new Date(c.at).toLocaleTimeString()}</span>
          </li>
        ))}
      </ul>
      <div className="flex items-center gap-2">
        <input
          className="flex-1 rounded-md border px-2 py-1 text-sm"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Add a comment"
        />
        <button
          className="px-2 py-1 rounded-md bg-gray-900 text-white text-xs"
          onClick={() => {
            const t = text.trim();
            if (!t) return;
            onAdd(t);
            setText("");
          }}
        >
          Reply
        </button>
      </div>
    </div>
  );
}

function PostCard({ post, users, currentUser, onLike, onAddComment }) {
  const author = users[post.author];
  const lastLikers = post.likes.slice(-5).reverse();
  const totalDelta = post.likes.reduce((a, e) => a + (e.delta || 0), 0);
  const uniqueLikers = new Set(post.likes.map((e) => e.from)).size;
  return (
    <div className="rounded-xl border p-3 bg-white">
      <div className="flex items-center gap-2 text-sm text-gray-500">
        <Avatar name={author.handle} />
        <div>
          @{author.handle} · <span className="opacity-70">#{post.id}</span>
        </div>
        <div className="ml-auto text-xs text-gray-500">Σδ={fmt(totalDelta, 2)} · {uniqueLikers} likers</div>
      </div>
      <div className="font-semibold mt-1">{post.title}</div>
      {post.image && (
        <img src={post.image} alt="" className="mt-2 w-full max-h-56 object-cover rounded-lg border" />
      )}
      <div className="text-sm text-gray-700 mt-2 whitespace-pre-wrap">{post.body}</div>
      {post.tags && post.tags.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-2">
          {post.tags.map((t, i) => (
            <span key={i} className="px-2 py-0.5 rounded-full bg-gray-100 text-xs">
              #{t}
            </span>
          ))}
        </div>
      )}
      <div className="flex items-center justify-between mt-3">
        <div className="text-xs text-gray-500 flex items-center gap-2">
          <span>likes: {post.likes.length}</span>
          <div className="flex -space-x-2">
            {lastLikers.map((lk, i) => (
              <div key={i} className="relative flex items-center">
                <div className="border-2 border-white rounded-full">
                  <Avatar name={users[lk.from].handle} />
                </div>
              </div>
            ))}
          </div>
        </div>
        <button
          onClick={() => onLike(currentUser, post.author, post.id)}
          className="px-3 py-1 rounded-md bg-black text-white text-sm hover:bg-gray-800"
        >
          Like as {users[currentUser].handle}
        </button>
      </div>
      {post.likes.length > 0 && (
        <div className="mt-3 border-t pt-2">
          <div className="text-xs text-gray-500 mb-1">Recent like log</div>
          <ul className="text-xs space-y-1">
            {lastLikers.map((lk, i) => (
              <li key={i} className="flex justify-between">
                <span>
                  @{users[lk.from].handle} → @{users[lk.to].handle}
                </span>
                <span>
                  δ={fmt(lk.delta, 2)}, α={fmt(lk.alpha, 3)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
      <CommentList post={post} users={users} onAdd={(text) => onAddComment(post.id, text)} />
    </div>
  );
}

function PostsList({ posts, users, currentUser, onLike, delta, onAddComment }) {
  return (
    <div className="grid md:grid-cols-2 gap-4">
      {posts.map((p) => (
        <PostCard
          key={p.id}
          post={p}
          users={users}
          currentUser={currentUser}
          onLike={onLike}
          delta={delta}
          onAddComment={onAddComment}
        />
      ))}
    </div>
  );
}

function LedgerTable({ ledger, users, posts }) {
  if (ledger.length === 0) return <div className="text-sm text-gray-500">No likes yet.</div>;
  return (
    <div className="overflow-auto">
      <table className="min-w-max text-right text-sm">
        <thead>
          <tr>
            <th className="px-2 py-1 text-left">time</th>
            <th className="px-2 py-1 text-left">from → to</th>
            <th className="px-2 py-1 text-left">post</th>
            <th className="px-2 py-1">δ</th>
            <th className="px-2 py-1">α</th>
          </tr>
        </thead>
        <tbody>
          {ledger
            .slice()
            .reverse()
            .slice(0, 100)
            .map((e) => (
              <tr key={e.id} className="border-t">
                <td className="px-2 py-1 text-left">{new Date(e.at).toLocaleTimeString()}</td>
                <td className="px-2 py-1 text-left">
                  @{users[e.from].handle} → @{users[e.to].handle}
                </td>
                <td className="px-2 py-1 text-left">{posts.find((p) => p.id === e.postId)?.title ?? `#${e.postId}`}</td>
                <td className="px-2 py-1">{fmt(e.delta, 3)}</td>
                <td className="px-2 py-1">{fmt(e.alpha, 3)}</td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------- Main App ----------
export default function App() {
  // ---- Initial state ----
  const [users, setUsers] = useState([
    { id: 0, handle: "alice" },
    { id: 1, handle: "bob" },
    { id: 2, handle: "carol" },
  ]);

  const makePost = (id, author, title, body, image = "", tags = []) => ({
    id,
    author,
    title,
    body,
    image,
    tags,
    likes: [],
    comments: [],
    createdAt: nowMs(),
  });

  const [posts, setPosts] = useState([
    makePost(
      1,
      0,
      "On PICSY",
      `Foundations & intuition.

– virtual central bank
– eigenvectors`,
      "",
      ["math", "picsy"]
    ),
    makePost(
      2,
      1,
      "Eigenvectors 101",
      `Left vs right.
Power iteration demo.`,
      "",
      ["linear-algebra"]
    ),
    makePost(3, 2, "Natural Recovery", "Gamma schedules explained.", "", ["recovery"]),
  ]);

  // 3x3: diag=0.2, off=0.4
  const initialE = useMemo(() => {
    const n = 3;
    const E = zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) E[i][j] = i === j ? 0.2 : 0.4;
    return ensureRowStochastic(E);
  }, []);

  const [E, setE] = useState(initialE);
  const [c, setC] = useState(() => powerIterationLeft(initialE));

  // Controls
  const [currentUser, setCurrentUser] = useState(0);
  const [delta, setDelta] = useState(0.05);
  const [gamma, setGamma] = useState(0.1);
  const [message, setMessage] = useState("");

  // Compose
  const [newTitle, setNewTitle] = useState("");
  const [newBody, setNewBody] = useState("");
  const [newImage, setNewImage] = useState("");
  const [newTags, setNewTags] = useState("");

  // Search / Sort
  const [q, setQ] = useState("");
  const [tagFilter, setTagFilter] = useState("");
  const [sortMode, setSortMode] = useState("new"); // new | hot | author-c

  // Simulation controls
  const [simOn, setSimOn] = useState(false);
  const [simIntervalMs, setSimIntervalMs] = useState(1500);
  const [simDeltaMin, setSimDeltaMin] = useState(0.02);
  const [simDeltaMax, setSimDeltaMax] = useState(0.08);

  // Derived
  const n = E.length;
  const budgets = useMemo(() => E.map((row, i) => row[i]), [E]);
  const pp = useMemo(() => budgets.map((b, i) => b * c[i]), [budgets, c]);
  const likesAvailable = useMemo(() => {
    const cost = delta / c[currentUser];
    return Math.max(0, Math.floor(E[currentUser][currentUser] / cost));
  }, [E, c, currentUser, delta]);

  // Ledger
  const [ledger, setLedger] = useState([]); // {id, from, to, postId, delta, alpha, at}
  const nextLedgerId = useRef(1);

  // E' preview (explicit)
  const EPrimePreview = useMemo(() => {
    const n = E.length;
    if (n <= 1) return [[1]];
    const out = zeros(n, n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) out[i][j] = i === j ? 0 : E[i][j] + E[i][i] / (n - 1);
    return out;
  }, [E]);

  // Trending score for posts (exponential decay on δ)
  const trendingScore = useMemo(() => {
    const tauMs = 30 * 60 * 1000; // 30min
    const now = nowMs();
    const scores = new Map();
    for (const p of posts) scores.set(p.id, 0);
    for (const e of ledger) {
      const age = Math.max(0, now - e.at);
      const w = Math.exp(-age / tauMs);
      scores.set(e.postId, (scores.get(e.postId) || 0) + w * e.delta);
    }
    return (postId) => scores.get(postId) || 0;
  }, [ledger, posts]);

  // Filters & sorting
  const visiblePosts = useMemo(() => {
    let arr = posts.slice();
    if (q.trim()) {
      const qq = q.toLowerCase();
      arr = arr.filter((p) => p.title.toLowerCase().includes(qq) || p.body.toLowerCase().includes(qq));
    }
    if (tagFilter.trim()) {
      arr = arr.filter((p) => p.tags.map((t) => t.toLowerCase()).includes(tagFilter.toLowerCase()));
    }
    if (sortMode === "new") arr.sort((a, b) => b.createdAt - a.createdAt);
    else if (sortMode === "hot") arr.sort((a, b) => trendingScore(b.id) - trendingScore(a.id));
    else if (sortMode === "author-c") arr.sort((a, b) => c[b.author] - c[a.author]);
    return arr;
  }, [posts, q, tagFilter, sortMode, trendingScore, c]);

  // Like flow matrix (Σδ)
  const likeFlow = useMemo(() => {
    const N = users.length;
    const M = zeros(N, N);
    for (const e of ledger) M[e.from][e.to] += e.delta;
    return M;
  }, [ledger, users.length]);

  // Helpers
  const recordLike = (b, s, postId, deltaVal, alphaVal) => {
    setPosts((ps) =>
      ps.map((p) => (p.id === postId ? { ...p, likes: [...p.likes, { from: b, to: s, delta: deltaVal, alpha: alphaVal, at: nowMs() }] } : p))
    );
    setLedger((lg) => [
      ...lg,
      { id: nextLedgerId.current++, from: b, to: s, postId, delta: deltaVal, alpha: alphaVal, at: nowMs() },
    ]);
  };

  const handleAddComment = (postId, text) => {
    const author = currentUser;
    setPosts((ps) =>
      ps.map((p) => (p.id === postId ? { ...p, comments: [...p.comments, { id: nowMs(), author, text, at: nowMs() }] } : p))
    );
  };

  // Handlers
  const handleLike = (b, s, postId) => {
    try {
      if (b === s) {
        setMessage("自身の投稿にはいいねできません。");
        return;
      }
      const alpha = delta / c[b];
      if (alpha > E[b][b] + 1e-12) {
        setMessage("予算不足で実行できません。");
        return;
      }
      const E2 = applyLike(E, c, b, s, delta);
      const c2 = powerIterationLeft(E2, { warmStart: normalizeToSum(c.map((x) => x / n), 1) });
      setE(E2);
      setC(c2);
      recordLike(b, s, postId, delta, alpha);
      setMessage(`✔ Like success: δ=${fmt(delta)} to @${users[s].handle} (α=${fmt(alpha)} by @${users[b].handle}).`);
    } catch (e) {
      setMessage(`✖ ${e.message}`);
    }
  };

  const handleRecovery = () => {
    const E2 = applyNaturalRecovery(E, clamp(gamma, 0.0, 0.99));
    const c2 = powerIterationLeft(E2, { warmStart: normalizeToSum(c.map((x) => x / n), 1) });
    setE(E2);
    setC(c2);
    setMessage(`↺ Recovery applied (γ=${fmt(gamma, 2)}).`);
  };

  const handleAddMember = () => {
    if (users.length >= 30) {
      setMessage("定員（30）に達しています。");
      return;
    }
    const newHandle = `user${users.length + 1}`;
    const { E: E2, c: c2 } = addMember(E, c, newHandle);
    setE(E2);
    setC(c2);
    setUsers((us) => [...us, { id: us.length, handle: newHandle }]);
    setPosts((ps) => [
      makePost(
        Math.max(0, ...ps.map((p) => p.id)) + 1,
        users.length,
        `${newHandle}'s first post`,
        "Hello PICSY!",
        "",
        ["hello"]
      ),
      ...ps,
    ]);
    setMessage(`＋ Added ${newHandle}. Existing budgets & contributions preserved.`);
  };

  const handleCreatePost = () => {
    const title = newTitle.trim() || "Untitled";
    const body = newBody.trim();
    const img = newImage.trim();
    const tags = newTags
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);
    setPosts((ps) => [makePost(Math.max(0, ...ps.map((p) => p.id)) + 1, currentUser, title, body, img, tags), ...ps]);
    setNewTitle("");
    setNewBody("");
    setNewImage("");
    setNewTags("");
    setMessage("✚ Post created.");
  };

  // Random like simulator
  useEffect(() => {
    if (!simOn) return;
    const id = setInterval(() => {
      const N = users.length;
      if (N < 2) return;
      let b = Math.floor(Math.random() * N);
      let s = Math.floor(Math.random() * N);
      if (N > 1) while (s === b) s = Math.floor(Math.random() * N);
      const candidates = posts.filter((p) => p.author === s);
      if (candidates.length === 0) return;
      const post = candidates[Math.floor(Math.random() * candidates.length)];
      const c_b = c[b];
      const Ebb = E[b][b];
      const delta0 = rnd(simDeltaMin, simDeltaMax);
      const maxDelta = Ebb * c_b;
      const deltaEff = Math.max(0, Math.min(delta0, maxDelta));
      if (deltaEff < 1e-6) return;
      try {
        const E2 = applyLike(E, c, b, s, deltaEff);
        const c2 = powerIterationLeft(E2, { warmStart: normalizeToSum(c.map((x) => x / n), 1) });
        setE(E2);
        setC(c2);
        const alpha = deltaEff / c[b];
        setPosts((ps) =>
          ps.map((p) => (p.id === post.id ? { ...p, likes: [...p.likes, { from: b, to: s, delta: deltaEff, alpha, at: nowMs() }] } : p))
        );
        setLedger((lg) => [
          ...lg,
          { id: nextLedgerId.current++, from: b, to: s, postId: post.id, delta: deltaEff, alpha, at: nowMs() },
        ]);
      } catch {
        /* ignore */
      }
    }, simIntervalMs);
    return () => clearInterval(id);
  }, [simOn, simIntervalMs, users.length, posts, E, c, simDeltaMin, simDeltaMax, n]);

  // Auto-clear messages after a few seconds
  useEffect(() => {
    if (!message) return;
    const t = setTimeout(() => setMessage(""), 5000);
    return () => clearTimeout(t);
  }, [message]);

  // Run self tests once on mount
  useEffect(() => {
    try {
      runSelfTests();
    } catch (e) {
      console.warn("Self-tests threw", e);
    }
  }, []);

  // Profile view derived stats
  const [profileIdx, setProfileIdx] = useState(0);
  const profileStats = useMemo(() => {
    const i = profileIdx;
    const handle = users[i]?.handle ?? "";
    const sent = ledger.filter((e) => e.from === i);
    const recv = ledger.filter((e) => e.to === i);
    const postsBy = posts.filter((p) => p.author === i);
    const sumSent = sent.reduce((a, e) => a + e.delta, 0);
    const sumRecv = recv.reduce((a, e) => a + e.delta, 0);
    const postRecv = postsBy.reduce((a, p) => a + p.likes.reduce((s, e) => s + e.delta, 0), 0);
    return {
      handle,
      c: c[i] || 0,
      budget: E[i]?.[i] || 0,
      pp: (E[i]?.[i] || 0) * (c[i] || 0),
      posts: postsBy.length,
      sumSent,
      sumRecv,
      postRecv,
    };
  }, [profileIdx, users, ledger, posts, c, E]);

  // Like flow matrix toggle
  const [showLikeFlow, setShowLikeFlow] = useState(true);

  // ---- UI ----
  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <div className="mx-auto max-w-7xl p-4 md:p-8">
        <header className="mb-6 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          <div>
            <h1 className="text-2xl font-bold">PICSY-Note Prototype</h1>
            <p className="text-sm text-gray-600">Virtual Central Bank · N ≤ 30 · Live random likes</p>
          </div>
          <div className="flex items-center gap-3">
            <div className="text-sm text-gray-600">
              Users: <span className="font-semibold">{users.length}</span>/30
            </div>
            <button onClick={handleAddMember} className="px-4 py-2 rounded-xl bg-indigo-600 text-white hover:bg-indigo-500 text-sm">
              Add Member
            </button>
          </div>
        </header>

        {message && <div className="mb-4 rounded-xl border bg-white p-3 text-sm">{message}</div>}

        <div className="grid lg:grid-cols-5 gap-6">
          {/* Left column */}
          <div className="lg:col-span-3 space-y-6">
            <Section
              title="Compose Post"
              actions={
                <div className="flex items-center gap-3 text-sm">
                  <label className="flex items-center gap-2">
                    <span>As</span>
                    <select className="rounded-md border px-2 py-1" value={currentUser} onChange={(e) => setCurrentUser(Number(e.target.value))}>
                      {users.map((u) => (
                        <option key={u.id} value={u.id}>
                          @{u.handle}
                        </option>
                      ))}
                    </select>
                  </label>
                  <span className="text-gray-500">likes available ≈ {likesAvailable}</span>
                  <span className="text-gray-500">α≈ {fmt(delta / c[currentUser], 4)}</span>
                </div>
              }
            >
              <div className="grid gap-2">
                <input value={newTitle} onChange={(e) => setNewTitle(e.target.value)} className="rounded-md border px-2 py-1" placeholder="Title" />
                <textarea value={newBody} onChange={(e) => setNewBody(e.target.value)} className="rounded-md border px-2 py-1" rows={4} placeholder="Body (supports line breaks)" />
                <input value={newImage} onChange={(e) => setNewImage(e.target.value)} className="rounded-md border px-2 py-1" placeholder="Image URL (optional)" />
                <input value={newTags} onChange={(e) => setNewTags(e.target.value)} className="rounded-md border px-2 py-1" placeholder="Tags comma-separated (optional)" />
                <div className="flex items-center gap-2">
                  <button onClick={handleCreatePost} className="px-3 py-1 rounded-md bg-gray-900 text-white hover:bg-gray-800 text-sm">
                    Publish
                  </button>
                  <div className="flex items-center gap-2 text-sm">
                    <span>δ</span>
                    <input type="number" step="0.01" min="0.01" max="0.50" value={delta} onChange={(e) => setDelta(clamp(Number(e.target.value), 0.01, 0.5))} className="w-24 rounded-md border px-2 py-1" />
                  </div>
                </div>
              </div>
            </Section>

            <Section
              title="Search & Sort"
              actions={
                <div className="flex items-center gap-2 text-sm">
                  <input className="rounded-md border px-2 py-1" placeholder="Search text" value={q} onChange={(e) => setQ(e.target.value)} />
                  <input className="rounded-md border px-2 py-1" placeholder="Filter tag" value={tagFilter} onChange={(e) => setTagFilter(e.target.value)} />
                  <select className="rounded-md border px-2 py-1" value={sortMode} onChange={(e) => setSortMode(e.target.value)}>
                    <option value="new">New</option>
                    <option value="hot">Hot (recent δ)</option>
                    <option value="author-c">Author c</option>
                  </select>
                </div>
              }
            >
              <div className="text-sm text-gray-600">検索語/タグで絞り込み、並び替えを選べます。Hotは近時のδを指数減衰で加重。</div>
            </Section>

            <Section title="Feed">
              <PostsList posts={visiblePosts} users={users} currentUser={currentUser} onLike={handleLike} delta={delta} onAddComment={handleAddComment} />
            </Section>

            <Section
              title="Random Like Simulator"
              actions={
                <div className="flex items-center gap-2 text-sm">
                  <label className="flex items-center gap-2">
                    <input type="checkbox" checked={simOn} onChange={(e) => setSimOn(e.target.checked)} />
                    <span>Enable</span>
                  </label>
                  <label className="flex items-center gap-1">
                    <span>interval(ms)</span>
                    <input type="number" className="w-24 rounded-md border px-2 py-1" value={simIntervalMs} onChange={(e) => setSimIntervalMs(clamp(Number(e.target.value), 250, 10000))} />
                  </label>
                  <label className="flex items-center gap-1">
                    <span>δ min</span>
                    <input type="number" step="0.01" className="w-20 rounded-md border px-2 py-1" value={simDeltaMin} onChange={(e) => setSimDeltaMin(clamp(Number(e.target.value), 0.005, 0.5))} />
                  </label>
                  <label className="flex items-center gap-1">
                    <span>δ max</span>
                    <input type="number" step="0.01" className="w-20 rounded-md border px-2 py-1" value={simDeltaMax} onChange={(e) => setSimDeltaMax(clamp(Number(e.target.value), simDeltaMin, 0.5))} />
                  </label>
                </div>
              }
              footer={"他メンバーが自動で価値を移転（いいね）します。予算が不足する場合はスキップします。"}
            >
              <div className="text-sm text-gray-600">乱択で buyer≠seller を選び、投稿を1つ選択。δは[min,max]から選び、α=δ/c_b が予算内なら適用します。</div>
            </Section>

            <Section
              title="Recovery"
              actions={
                <div className="flex items-center gap-2 text-sm">
                  <label className="flex items-center gap-2">
                    <span>γ</span>
                    <input type="number" step="0.01" min="0.01" max="0.50" value={gamma} onChange={(e) => setGamma(clamp(Number(e.target.value), 0.01, 0.5))} className="w-24 rounded-md border px-2 py-1" />
                  </label>
                  <button onClick={handleRecovery} className="px-3 py-1 rounded-md bg-gray-900 text-white hover:bg-gray-800">
                    Apply Recovery
                  </button>
                </div>
              }
            >
              <div className="text-sm text-gray-600">Off-diagonal × (1−γ) / Diagonal ← Diagonal + γ(1−Diagonal)</div>
            </Section>
          </div>

          {/* Right column */}
          <div className="lg:col-span-2 space-y-6">
            <Section title="Contributions (c) & Budgets & PP">
              <div className="grid grid-cols-1 gap-4">
                <VectorTable title="c (sum = N)" vec={c} />
                <VectorTable title="budgets (diag E)" vec={budgets} />
                <VectorTable title="PP = budget × c" vec={pp} />
              </div>
            </Section>

            <Section title="Matrices">
              <div className="grid grid-cols-1 gap-4">
                <MatrixTable title="E (evaluation matrix)" matrix={E} highlightDiag c={c} pp={pp} />
                <MatrixTable title="E' (effective)" matrix={EPrimePreview} />
              </div>
            </Section>

            <Section title="Global Like Ledger">
              <LedgerTable ledger={ledger} users={users} posts={posts} />
            </Section>

            <Section title="Profile">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm">User</span>
                <select className="rounded-md border px-2 py-1 text-sm" value={profileIdx} onChange={(e) => setProfileIdx(Number(e.target.value))}>
                  {users.map((u, i) => (
                    <option key={u.id} value={i}>
                      @{u.handle}
                    </option>
                  ))}
                </select>
              </div>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  c: <span className="font-semibold">{fmt(profileStats.c)}</span>
                </div>
                <div>
                  budget: <span className="font-semibold">{fmt(profileStats.budget)}</span>
                </div>
                <div>
                  PP: <span className="font-semibold">{fmt(profileStats.pp)}</span>
                </div>
                <div>
                  posts: <span className="font-semibold">{profileStats.posts}</span>
                </div>
                <div>
                  δ received (posts): <span className="font-semibold">{fmt(profileStats.postRecv, 2)}</span>
                </div>
                <div>
                  δ received (direct): <span className="font-semibold">{fmt(profileStats.sumRecv, 2)}</span>
                </div>
                <div>
                  δ sent: <span className="font-semibold">{fmt(profileStats.sumSent, 2)}</span>
                </div>
              </div>
            </Section>

            <Section
              title="Who-Liked-Whom (Σδ)"
              actions={
                <label className="text-sm flex items-center gap-2">
                  <input type="checkbox" checked={showLikeFlow} onChange={(e) => setShowLikeFlow(e.target.checked)} />
                  Show
                </label>
              }
            >
              {showLikeFlow ? (
                <MatrixTable title="Like Flow Σδ (rows=buyer, cols=seller)" matrix={likeFlow} />
              ) : (
                <div className="text-sm text-gray-500">hidden</div>
              )}
            </Section>
          </div>
        </div>

        <footer className="mt-10 text-xs text-gray-500">
          <div>
            Math: δ = α·c_b ⇒ α = δ / c_b. c is the left eigenvector of E' (sum = N). Add member uses x = 1/N; new row = c/N; new col = x(1−E_ii).
          </div>
        </footer>
      </div>
    </div>
  );
}
