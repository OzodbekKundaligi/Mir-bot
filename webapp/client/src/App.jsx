import { useEffect, useMemo, useState } from "react";
import {
  addComment,
  buildMediaUrl,
  fetchBootstrap,
  fetchComments,
  fetchContent,
  fetchContentDetail,
  fetchFavorites,
  fetchProfile,
  fetchRecommendations,
  getTelegramInitData,
  setReaction,
  toggleFavorite,
  trackDownload
} from "./api";

const TAB = {
  HOME: "home",
  FAVORITES: "favorites",
  HISTORY: "history",
  PROFILE: "profile"
};

function openTelegramLink(url) {
  const tg = window.Telegram?.WebApp;
  if (tg?.openTelegramLink) {
    tg.openTelegramLink(url);
    return;
  }
  window.open(url, "_blank", "noopener,noreferrer");
}

function SubscriptionGate({ missingChannels, onRefresh }) {
  return (
    <section className="gate">
      <h2>Majburiy obuna kerak</h2>
      <p>Platformani ishlatishdan oldin quyidagi kanallarga obuna bo'ling.</p>
      <div className="channel-list">
        {missingChannels.map((channel) => (
          <div className="channel-item" key={channel.channel_ref}>
            <div>
              <strong>{channel.title}</strong>
              <small>{channel.channel_ref}</small>
            </div>
            {channel.join_url ? (
              <button onClick={() => openTelegramLink(channel.join_url)}>Qo'shilish</button>
            ) : (
              <span>Link yo'q</span>
            )}
          </div>
        ))}
      </div>
      <button className="primary" onClick={onRefresh}>
        Obunani tekshirish
      </button>
    </section>
  );
}

function ContentCard({ item, initData, isFavorite, onOpen, onToggleFavorite, onDownload }) {
  const previewUrl = item.preview_file_id ? buildMediaUrl(item.preview_file_id, initData) : "";
  return (
    <article className="card">
      <button className="thumb thumb-btn" onClick={() => onOpen(item)}>
        {previewUrl ? <img src={previewUrl} alt={item.title} loading="lazy" /> : <div className="thumb-fallback">PLAY</div>}
      </button>
      <div className="card-body">
        <h3>{item.title || "Noma'lum kontent"}</h3>
        <p>{item.description || "Tavsif yo'q"}</p>
        <div className="meta">
          <span>{item.code || "-"}</span>
          <span>{item.year || "-"}</span>
          <span>{item.downloads || 0} yuklash</span>
        </div>
        <div className="engagement-row">
          <small>{item.likes || 0} like</small>
          <small>{item.dislikes || 0} dislike</small>
          <small>{item.comments || 0} comment</small>
        </div>
        <div className="actions">
          <button onClick={() => onToggleFavorite(item)}>{isFavorite ? "Saqlangan" : "Saqlash"}</button>
          <button className="ghost" onClick={() => onDownload(item)}>
            Yuklab olish
          </button>
          <button className="ghost" onClick={() => onOpen(item)}>
            Ochish
          </button>
        </div>
      </div>
    </article>
  );
}

function DetailDrawer({
  open,
  item,
  comments,
  recommendations,
  initData,
  commentText,
  setCommentText,
  onClose,
  onReact,
  onAddComment,
  onToggleFavorite,
  onDownload,
  onOpenRecommendation,
  isFavorite
}) {
  if (!open || !item) return null;
  const previewUrl = item.preview_file_id ? buildMediaUrl(item.preview_file_id, initData) : "";
  return (
    <section className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-head">
          <h2>{item.title}</h2>
          <button className="ghost" onClick={onClose}>
            Yopish
          </button>
        </div>
        {previewUrl ? <img className="drawer-preview" src={previewUrl} alt={item.title} /> : null}
        <p className="drawer-desc">{item.description || "Tavsif yo'q"}</p>
        <div className="drawer-actions">
          <button className={item.user_reaction === "like" ? "active" : ""} onClick={() => onReact("like")}>
            Like ({item.likes || 0})
          </button>
          <button className={item.user_reaction === "dislike" ? "active" : ""} onClick={() => onReact("dislike")}>
            Yoqmadi ({item.dislikes || 0})
          </button>
          <button onClick={() => onToggleFavorite(item)}>{isFavorite ? "Saqlangan" : "Saqlash"}</button>
          <button className="ghost" onClick={() => onDownload(item)}>
            Yuklab olish
          </button>
        </div>
        <div className="comment-box">
          <h4>Commentlar ({item.comments || 0})</h4>
          <div className="comment-write">
            <textarea
              value={commentText}
              onChange={(e) => setCommentText(e.target.value)}
              placeholder="Fikringizni yozing..."
            />
            <button onClick={onAddComment}>Yuborish</button>
          </div>
          <div className="comment-list">
            {comments.map((comment) => (
              <div className="comment-item" key={comment.id}>
                <strong>{comment.full_name || "User"}</strong>
                <small>{comment.username ? `@${comment.username}` : "user"}</small>
                <p>{comment.text}</p>
              </div>
            ))}
            {comments.length === 0 ? <p className="muted">Hozircha comment yo'q.</p> : null}
          </div>
        </div>
        <div className="recommend-box">
          <h4>O'xshash kontent</h4>
          <div className="recommend-list">
            {recommendations.map((rec) => (
              <button key={`${rec.content_type}:${rec.id}`} className="recommend-item" onClick={() => onOpenRecommendation(rec)}>
                <strong>{rec.title}</strong>
                <small>{rec.year || "-"} | {rec.likes || 0} like</small>
              </button>
            ))}
            {recommendations.length === 0 ? <p className="muted">Hozircha tavsiya topilmadi.</p> : null}
          </div>
        </div>
      </aside>
    </section>
  );
}

export default function App() {
  const [initData, setInitData] = useState(getTelegramInitData());
  const [activeTab, setActiveTab] = useState(TAB.HOME);
  const [query, setQuery] = useState("");
  const [contentType, setContentType] = useState("all");
  const [loading, setLoading] = useState(true);
  const [searchLoading, setSearchLoading] = useState(false);
  const [error, setError] = useState("");
  const [boot, setBoot] = useState(null);
  const [content, setContent] = useState([]);
  const [favorites, setFavorites] = useState([]);
  const [history, setHistory] = useState([]);
  const [trending, setTrending] = useState([]);
  const [profile, setProfile] = useState(null);
  const [detailOpen, setDetailOpen] = useState(false);
  const [detailItem, setDetailItem] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [comments, setComments] = useState([]);
  const [commentText, setCommentText] = useState("");

  useEffect(() => {
    const tg = window.Telegram?.WebApp;
    if (tg) {
      tg.ready();
      tg.expand();
      tg.setHeaderColor("#0f1724");
      const data = tg.initData || "";
      if (data) {
        setInitData(data);
      }
    }
  }, []);

  function patchItemEverywhere(contentTypeValue, contentRef, patch) {
    const applyPatch = (arr) =>
      arr.map((row) => {
        if (row.content_type === contentTypeValue && row.id === contentRef) {
          return { ...row, ...patch };
        }
        return row;
      });
    setContent((prev) => applyPatch(prev));
    setFavorites((prev) => applyPatch(prev));
    setHistory((prev) => applyPatch(prev));
    setTrending((prev) => applyPatch(prev));
    setDetailItem((prev) => {
      if (!prev) return prev;
      if (prev.content_type === contentTypeValue && prev.id === contentRef) {
        return { ...prev, ...patch };
      }
      return prev;
    });
  }

  async function loadBootstrap() {
    if (!initData) {
      setLoading(false);
      setError("Telegram ichidan oching: botdagi Web ilova tugmasini bosing.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const data = await fetchBootstrap(initData);
      setBoot(data);
      setContent(data.content || []);
      setFavorites(data.favorites || []);
      setHistory(data.history || []);
      setTrending(data.trending || []);
      setProfile({ user: data.user, stats: data.stats, history: data.history || [] });
    } catch (err) {
      setError(err.message || "Yuklash xatosi");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadBootstrap();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initData]);

  useEffect(() => {
    if (!initData || !boot || boot.blocked) return;
    const abort = new AbortController();
    const timer = setTimeout(async () => {
      setSearchLoading(true);
      try {
        const data = await fetchContent({
          initData,
          query,
          contentType,
          signal: abort.signal
        });
        setContent(data.items || []);
      } catch (err) {
        if (abort.signal.aborted) return;
        if (err.status === 403) {
          await loadBootstrap();
        } else {
          setError(err.message || "Qidiruv xatosi");
        }
      } finally {
        setSearchLoading(false);
      }
    }, 250);
    return () => {
      abort.abort();
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, contentType, initData, boot?.blocked]);

  const favoriteMap = useMemo(() => {
    const map = new Set();
    for (const row of favorites) {
      map.add(`${row.content_type}:${row.id}`);
    }
    return map;
  }, [favorites]);

  async function onToggleFavorite(item) {
    if (!initData) return;
    try {
      await toggleFavorite({
        initData,
        contentType: item.content_type,
        contentRef: item.id
      });
      const favData = await fetchFavorites(initData);
      setFavorites(favData.items || []);
    } catch (err) {
      setError(err.message || "Saqlash xatosi");
    }
  }

  async function onDownload(item) {
    if (!initData) return;
    try {
      const result = await trackDownload({
        initData,
        contentType: item.content_type,
        contentRef: item.id
      });
      if (item.deep_link) {
        openTelegramLink(item.deep_link);
      }
      patchItemEverywhere(item.content_type, item.id, result.summary || { downloads_tracked: (item.downloads_tracked || 0) + 1 });
    } catch (err) {
      setError(err.message || "Yuklab olish xatosi");
    }
  }

  async function openDetail(item) {
    if (!initData) return;
    setDetailOpen(true);
    setCommentText("");
    try {
      const [detailData, commentsData] = await Promise.all([
        fetchContentDetail({
          initData,
          contentType: item.content_type,
          contentRef: item.id
        }),
        fetchComments({
          initData,
          contentType: item.content_type,
          contentRef: item.id
        })
      ]);
      let recItems = detailData.recommendations || [];
      try {
        const recData = await fetchRecommendations({
          initData,
          contentType: item.content_type,
          contentRef: item.id,
          limit: 10
        });
        recItems = recData.items || recItems;
      } catch {
        recItems = detailData.recommendations || [];
      }
      setDetailItem(detailData.item);
      setComments(commentsData.items || []);
      setRecommendations(recItems);
      patchItemEverywhere(item.content_type, item.id, detailData.item);
      setHistory((prev) => {
        const filtered = prev.filter((row) => !(row.content_type === detailData.item.content_type && row.id === detailData.item.id));
        return [detailData.item, ...filtered].slice(0, 40);
      });
    } catch (err) {
      setError(err.message || "Detail xatosi");
      setDetailOpen(false);
    }
  }

  async function onReact(reaction) {
    if (!initData || !detailItem) return;
    const desired = detailItem.user_reaction === reaction ? "none" : reaction;
    try {
      const result = await setReaction({
        initData,
        contentType: detailItem.content_type,
        contentRef: detailItem.id,
        reaction: desired
      });
      patchItemEverywhere(detailItem.content_type, detailItem.id, result.summary);
    } catch (err) {
      setError(err.message || "Reaction xatosi");
    }
  }

  async function onAddComment() {
    if (!initData || !detailItem) return;
    const text = commentText.trim();
    if (!text) return;
    try {
      const result = await addComment({
        initData,
        contentType: detailItem.content_type,
        contentRef: detailItem.id,
        text
      });
      setComments((prev) => [result.item, ...prev]);
      setCommentText("");
      patchItemEverywhere(detailItem.content_type, detailItem.id, result.summary || { comments: (detailItem.comments || 0) + 1 });
    } catch (err) {
      setError(err.message || "Comment qo'shishda xato");
    }
  }

  async function openRecommendation(item) {
    await openDetail(item);
  }

  async function openProfileTab() {
    setActiveTab(TAB.PROFILE);
    if (!initData) return;
    try {
      const data = await fetchProfile(initData);
      setProfile(data);
    } catch (err) {
      setError(err.message || "Profil xatosi");
    }
  }

  const blocked = Boolean(boot?.blocked);

  return (
    <main className="page">
      <div className="bg-shape bg-1" />
      <div className="bg-shape bg-2" />
      <header className="topbar">
        <div>
          <h1>KinoTube</h1>
          <p>Yangi avlod Telegram media platformasi</p>
        </div>
      </header>

      {loading ? <section className="panel">Yuklanmoqda...</section> : null}
      {error ? <section className="panel error">{error}</section> : null}

      {!loading && blocked ? (
        <SubscriptionGate missingChannels={boot?.missing_channels || []} onRefresh={loadBootstrap} />
      ) : null}

      {!loading && !blocked ? (
        <>
          <nav className="tabs tabs-4">
            <button className={activeTab === TAB.HOME ? "active" : ""} onClick={() => setActiveTab(TAB.HOME)}>
              Feed
            </button>
            <button className={activeTab === TAB.FAVORITES ? "active" : ""} onClick={() => setActiveTab(TAB.FAVORITES)}>
              Saqlangan
            </button>
            <button className={activeTab === TAB.HISTORY ? "active" : ""} onClick={() => setActiveTab(TAB.HISTORY)}>
              Tarix
            </button>
            <button className={activeTab === TAB.PROFILE ? "active" : ""} onClick={openProfileTab}>
              Profil
            </button>
          </nav>

          {activeTab === TAB.HOME ? (
            <section className="panel">
              <div className="section-head">
                <h2>Trendlar</h2>
                <small>Eng ko'p qiziqilgan kontent</small>
              </div>
              <div className="trend-row">
                {trending.slice(0, 8).map((item) => (
                  <button key={`${item.content_type}:${item.id}`} className="trend-pill" onClick={() => openDetail(item)}>
                    <strong>{item.title}</strong>
                    <small>{item.likes || 0} like</small>
                  </button>
                ))}
              </div>
              <div className="filters">
                <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Qidiruv: kino, serial, kod" />
                <select value={contentType} onChange={(e) => setContentType(e.target.value)}>
                  <option value="all">Barchasi</option>
                  <option value="movie">Kino</option>
                  <option value="serial">Serial</option>
                </select>
              </div>
              {searchLoading ? <p className="muted">Qidirilmoqda...</p> : null}
              <div className="grid">
                {content.map((item) => (
                  <ContentCard
                    key={`${item.content_type}:${item.id}`}
                    item={item}
                    initData={initData}
                    isFavorite={favoriteMap.has(`${item.content_type}:${item.id}`)}
                    onOpen={openDetail}
                    onToggleFavorite={onToggleFavorite}
                    onDownload={onDownload}
                  />
                ))}
              </div>
            </section>
          ) : null}

          {activeTab === TAB.FAVORITES ? (
            <section className="panel">
              <div className="grid">
                {favorites.map((item) => (
                  <ContentCard
                    key={`${item.content_type}:${item.id}`}
                    item={item}
                    initData={initData}
                    isFavorite
                    onOpen={openDetail}
                    onToggleFavorite={onToggleFavorite}
                    onDownload={onDownload}
                  />
                ))}
              </div>
            </section>
          ) : null}

          {activeTab === TAB.HISTORY ? (
            <section className="panel">
              <div className="grid">
                {history.map((item) => (
                  <ContentCard
                    key={`${item.content_type}:${item.id}`}
                    item={item}
                    initData={initData}
                    isFavorite={favoriteMap.has(`${item.content_type}:${item.id}`)}
                    onOpen={openDetail}
                    onToggleFavorite={onToggleFavorite}
                    onDownload={onDownload}
                  />
                ))}
              </div>
            </section>
          ) : null}

          {activeTab === TAB.PROFILE ? (
            <section className="panel">
              <h2>Profil</h2>
              <p>
                {(profile?.user?.first_name || "") + " " + (profile?.user?.last_name || "")}
                {profile?.user?.username ? ` (@${profile.user.username})` : ""}
              </p>
              <div className="stats">
                <div>
                  <strong>{profile?.stats?.movies || 0}</strong>
                  <span>Kino</span>
                </div>
                <div>
                  <strong>{profile?.stats?.serials || 0}</strong>
                  <span>Serial</span>
                </div>
                <div>
                  <strong>{profile?.stats?.favorites || 0}</strong>
                  <span>Saqlangan</span>
                </div>
                <div>
                  <strong>{profile?.stats?.comments || 0}</strong>
                  <span>Comment</span>
                </div>
                <div>
                  <strong>{profile?.stats?.likes_given || 0}</strong>
                  <span>Like</span>
                </div>
                <div>
                  <strong>{profile?.stats?.downloads || 0}</strong>
                  <span>Yuklab olgan</span>
                </div>
                <div>
                  <strong>{profile?.stats?.dislikes_given || 0}</strong>
                  <span>Dislike</span>
                </div>
                <div>
                  <strong>{profile?.stats?.history_views || 0}</strong>
                  <span>Ko'rishlar</span>
                </div>
              </div>
            </section>
          ) : null}
        </>
      ) : null}

      <DetailDrawer
        open={detailOpen}
        item={detailItem}
        comments={comments}
        recommendations={recommendations}
        initData={initData}
        commentText={commentText}
        setCommentText={setCommentText}
        onClose={() => {
          setDetailOpen(false);
          setRecommendations([]);
        }}
        onReact={onReact}
        onAddComment={onAddComment}
        onToggleFavorite={onToggleFavorite}
        onDownload={onDownload}
        onOpenRecommendation={openRecommendation}
        isFavorite={detailItem ? favoriteMap.has(`${detailItem.content_type}:${detailItem.id}`) : false}
      />
    </main>
  );
}
