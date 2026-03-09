import React, { useEffect, useMemo, useState } from "https://esm.sh/react@18";
import { createRoot } from "https://esm.sh/react-dom@18/client";
import htm from "https://esm.sh/htm@3.1.1";

const html = htm.bind(React.createElement);
const tg = window.Telegram?.WebApp ?? null;

if (tg) {
  try {
    tg.ready();
    tg.expand();
    tg.setHeaderColor("#070b16");
    tg.setBackgroundColor("#070b16");
  } catch (_) {}
}

const NAV = [
  { key: "home", label: "Bosh sahifa", icon: "home" },
  { key: "search", label: "Qidiruv", icon: "search" },
  { key: "saved", label: "Saqlangan", icon: "bookmark" },
  { key: "pro", label: "PRO", icon: "crown" },
  { key: "news", label: "Yangiliklar", icon: "spark" },
  { key: "profile", label: "Profil", icon: "user" },
];

const PRIMARY_NAV_KEYS = ["home", "search", "saved", "profile"];

const SEARCH_TYPES = [
  { key: "all", label: "Barchasi" },
  { key: "movie", label: "Kinolar" },
  { key: "serial", label: "Seriallar" },
];

const ADMIN_BOT_ACTIONS = [
  { action: "open_admin_panel", label: "Panel", icon: "shield" },
  { action: "admin_subs", label: "Obuna", icon: "grid" },
  { action: "admin_add_movie", label: "Kino qo'shish", icon: "play" },
  { action: "admin_add_serial", label: "Serial qo'shish", icon: "play" },
  { action: "admin_edit_content", label: "Tahrirlash", icon: "grid" },
  { action: "admin_delete_content", label: "O'chirish", icon: "close" },
  { action: "admin_list_content", label: "Baza", icon: "bookmark" },
  { action: "admin_broadcast", label: "Xabar", icon: "megaphone" },
  { action: "admin_requests", label: "So'rovlar", icon: "stats" },
  { action: "admin_stats", label: "Statistika", icon: "stats" },
  { action: "admin_add_admin", label: "Admin qo'shish", icon: "user" },
];

function icon(name) {
  const icons = {
    home: html`<path d="M4 10.5 12 4l8 6.5V20a1 1 0 0 1-1 1h-4.5v-6h-5v6H5a1 1 0 0 1-1-1z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>`,
    search: html`<path d="m20 20-4.2-4.2" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="10.5" cy="10.5" r="6.5" fill="none" stroke="currentColor" stroke-width="1.8"/>`,
    bookmark: html`<path d="M7 4h10a1 1 0 0 1 1 1v15l-6-3-6 3V5a1 1 0 0 1 1-1Z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>`,
    crown: html`<path d="m4 9 4.2 3L12 6l3.8 6L20 9l-1.5 10h-13Z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>`,
    megaphone: html`<path d="M14 6 20 4v16l-6-2.2V6Zm-1 1H8a3 3 0 0 0-3 3v3a3 3 0 0 0 3 3h5V7Zm-4 10 1.7 4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>`,
    user: html`<path d="M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm-7 8a7 7 0 0 1 14 0" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>`,
    shield: html`<path d="M12 3 5 6v5c0 4.6 2.7 7.8 7 10 4.3-2.2 7-5.4 7-10V6z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>`,
    stats: html`<path d="M5 19V9m7 10V5m7 14v-7" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="M3 21h18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>`,
    play: html`<path d="m9 7 8 5-8 5z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>`,
    close: html`<path d="m6 6 12 12M18 6 6 18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>`,
    link: html`<path d="M10 14 8 16a3 3 0 1 1-4.2-4.2l3-3A3 3 0 0 1 11 8m3 8 2-2a3 3 0 1 0-4.2-4.2l-1 1m-2 2h6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>`,
    grid: html`<path d="M4 4h7v7H4zm9 0h7v7h-7zM4 13h7v7H4zm9 0h7v7h-7z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>`,
    menu: html`<path d="M4 7h16M4 12h16M4 17h16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>`,
    spark: html`<path d="m12 3 1.8 4.7L19 9.4l-4 3.2 1.4 5-4.4-2.8-4.4 2.8 1.4-5-4-3.2 5.2-1.7L12 3Z" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/>`,
    arrowRight: html`<path d="M5 12h13m-5-5 5 5-5 5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>`,
    thumbsUp: html`<path d="M8 11v9H5a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1h3Zm3 9h4.8a2 2 0 0 0 2-1.7l1-6.5A2 2 0 0 0 16.8 9H13l.6-3.1c.2-1-.6-1.9-1.6-1.9h-.4L8 11" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>`,
    thumbsDown: html`<path d="M8 13V4H5a1 1 0 0 0-1 1v7a1 1 0 0 0 1 1h3Zm3-9h4.8a2 2 0 0 1 2 1.7l1 6.5a2 2 0 0 1-2 2.3H13l.6 3.1c.2 1-.6 1.9-1.6 1.9h-.4L8 13" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>`,
  };
  return html`<svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">${icons[name] || icons.grid}</svg>`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    method: options.method || "GET",
    headers: {
      "Content-Type": "application/json",
      "X-Telegram-Init-Data": tg?.initData || "",
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const payload = await response.json().catch(() => ({ ok: false, detail: "Javob formati xato" }));
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.detail || "So'rov bajarilmadi");
  }
  return payload;
}

function joinClass(...values) {
  return values.filter(Boolean).join(" ");
}

function compact(value) {
  const number = Number(value || 0);
  if (number >= 1000000) return `${(number / 1000000).toFixed(1)}M`;
  if (number >= 1000) return `${(number / 1000).toFixed(1)}K`;
  return String(number);
}

function dateText(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("uz-UZ", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
}

function dateValue(value) {
  if (!value) return 0;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 0 : date.getTime();
}

function statusText(value) {
  const lookup = {
    pending: "Kutilmoqda",
    approved: "Tasdiqlandi",
    rejected: "Rad etildi",
    posted: "Joylandi",
    active: "Faol",
    inactive: "Faol emas",
    free: "Oddiy",
  };
  const key = String(value || "").trim().toLowerCase();
  return lookup[key] || String(value || "—");
}

function yesNo(value) {
  return value ? "Ha" : "Yo'q";
}

function copyText(value, notify) {
  navigator.clipboard.writeText(String(value || "")).then(() => notify("Nusxa olindi")).catch(() => notify("Nusxa olib bo'lmadi"));
}

function buildBotActionLink(action, links) {
  const username = String(links?.bot_username || "").trim().replace(/^@/, "");
  const actionName = String(action || "").trim();
  if (!username || !actionName) return "";
  return `https://t.me/${username}?start=${encodeURIComponent(`wa_${actionName}`)}`;
}

function openTelegramTarget(url) {
  if (!url) return false;
  if (tg?.openTelegramLink) {
    tg.openTelegramLink(url);
    return true;
  }
  if (tg?.openLink) {
    tg.openLink(url);
    return true;
  }
  window.location.href = url;
  return true;
}

function openUrl(url) {
  const target = String(url || "").trim();
  if (!target) return;
  if (target.startsWith("https://t.me/")) {
    openTelegramTarget(target);
    return;
  }
  if (tg?.openLink) {
    tg.openLink(target);
    return;
  }
  window.open(target, "_blank", "noopener,noreferrer");
}

function sendToBot(payload, notify, links) {
  const action = String(payload?.action || "").trim();
  const deepLink = buildBotActionLink(action, links);
  if (deepLink) {
    openTelegramTarget(deepLink);
    return;
  }
  if (tg?.sendData) {
    tg.sendData(JSON.stringify(payload));
    notify("So'rov botga yuborildi");
    return;
  }
  notify("Ilovani Telegram ichidan oching");
}

function openInBot(item, notify, links) {
  const deepLink = String(item?.deep_link || "").trim();
  if (deepLink) {
    openTelegramTarget(deepLink);
    return;
  }
  sendToBot(item?.open_payload || {}, notify, links);
}

async function shareItem(item, notify, links) {
  const title = String(item?.title || "Mir Top Kino").trim();
  const url = String(item?.deep_link || links?.bot_url || "").trim();
  if (navigator.share && url) {
    try {
      await navigator.share({ title, text: title, url });
      return;
    } catch (_) {}
  }
  if (url) {
    copyText(url, notify);
    return;
  }
  copyText(item?.code || title, notify);
}

function Media({ item, detail = false }) {
  if (item?.preview_kind === "video" && item?.preview_url) {
    return html`<video className=${detail ? "detail-media" : "content-media"} src=${item.preview_url} muted loop autoPlay playsInline preload="metadata"></video>`;
  }
  if (item?.preview_url) {
    return html`<img className=${detail ? "detail-media" : "content-media"} src=${item.preview_url} alt=${item.title || "preview"} loading="lazy" />`;
  }
  return html`<div className=${detail ? "detail-media" : "content-media"} style=${{ display: "grid", placeItems: "center" }}><div className="pill">${item?.code || "MEDIA"}</div></div>`;
}

function Empty({ title, copy }) {
  return html`<div className="empty"><div style=${{ fontWeight: 700, marginBottom: "8px" }}>${title}</div><div>${copy}</div></div>`;
}

function SectionSarlavha({ iconName, title, copy, action }) {
  return html`<div className="section-header"><div><h2 className="section-title">${icon(iconName)}${title}</h2>${copy ? html`<p className="section-copy">${copy}</p>` : null}</div>${action || null}</div>`;
}

function initialsFromName(value) {
  const parts = String(value || "").trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "U";
  return parts.slice(0, 2).map((item) => item[0]?.toUpperCase() || "").join("");
}

function RailSection({ iconName, title, copy, items, onOpen, onFavorite, onReact, action }) {
  return html`<section className="section"><${SectionSarlavha} iconName=${iconName} title=${title} copy=${copy} action=${action} />${items?.length ? html`<div className="rail-track">${items.map((item) => html`<div className="rail-item" key=${`${item.content_type}-${item.id}`}><${PosterCard} item=${item} onOpen=${onOpen} onFavorite=${onFavorite} onReact=${onReact} /></div>`)}</div>` : html`<${Empty} title="Bo'sh" copy="Kontent shu yerda ko'rinadi." />`}</section>`;
}

function SkeletonGrid({ count = 6 }) {
  return html`<div className="content-grid">${Array.from({ length: count }).map((_, index) => html`<div className="skeleton-card" key=${index}><div className="skeleton-media"></div><div className="skeleton-line wide"></div><div className="skeleton-line"></div><div className="skeleton-actions"><div className="skeleton-chip"></div><div className="skeleton-chip"></div><div className="skeleton-chip"></div></div></div>`)}</div>`;
}

function Card({ item, onOpen, onFavorite, onReact }) {
  return html`<article className="content-card" onClick=${() => onOpen(item)}><${Media} item=${item} /><div className="card-hover-play">${icon("play")}</div><div className="content-card-body"><div className="content-card-top"><div><h3 className="content-title">${item.title || "Nomsiz"}</h3><div className="content-meta"><span>${item.code || "—"}</span><span>${item.year || "—"}</span><span>${item.quality || "HD"}</span></div></div><div className="tag">${item.content_type === "serial" ? "Serial" : "Kino"}</div></div><div className="content-meta"><span>${compact(item.views)} ko'rish</span><span>${compact(item.likes)} yoqdi</span><span>${Number(item.rating || 0).toFixed(1)} reyting</span></div><div className="content-actions compact" onClick=${(event) => event.stopPropagation()}><button className=${joinClass("action-pill", item.is_favorite && "active")} onClick=${() => onFavorite(item)} aria-label=${item.is_favorite ? "Saqlangandan olish" : "Saqlash"}>${icon("bookmark")}</button><button className=${joinClass("action-pill", item.user_reaction === "like" && "active")} onClick=${() => onReact(item, "like")} aria-label="Yoqdi">${icon("thumbsUp")}<span>${compact(item.likes)}</span></button><button className=${joinClass("action-pill", item.user_reaction === "dislike" && "active")} onClick=${() => onReact(item, "dislike")} aria-label="Yoqmadi">${icon("thumbsDown")}<span>${compact(item.dislikes)}</span></button><button className="action-pill primary" onClick=${() => onOpen(item)} aria-label="Ochish">${icon("play")}</button></div></div></article>`;
}

function PosterCard({ item, onOpen, onFavorite, onReact }) {
  return html`<article className="poster-card" onClick=${() => onOpen(item)}><${Media} item=${item} /><div className="poster-overlay"></div><div className="poster-body"><div className="poster-top"><h3 className="poster-title">${item.title || "Nomsiz"}</h3><span className="poster-year">${item.year || "—"}</span></div><div className="poster-meta"><span>${compact(item.views)} ko'rish</span><span>★ ${Number(item.rating || 0).toFixed(1)}</span></div><div className="poster-actions" onClick=${(event) => event.stopPropagation()}><button className=${joinClass("mini-action", item.is_favorite && "active")} onClick=${() => onFavorite(item)} aria-label="Saqlash">${icon("bookmark")}</button><button className=${joinClass("mini-action", item.user_reaction === "like" && "active")} onClick=${() => onReact(item, "like")} aria-label="Yoqdi">${icon("thumbsUp")}</button><button className="mini-action active" onClick=${() => onOpen(item)} aria-label="Ochish">${icon("play")}</button></div></div></article>`;
}

 function DetailSheet({ item, onClose, onFavorite, onReact, onBotOpen, onShare }) {
  if (!item) return null;
  return html`<div className="sheet-backdrop" onClick=${onClose}><div className="sheet" onClick=${(event) => event.stopPropagation()}><div className="sheet-header"><div><div className="eyebrow">${item.content_type === "serial" ? "Serial haqida" : "Kino haqida"}</div><h2 style=${{ margin: "8px 0 0", fontSize: "28px" }}>${item.title}</h2></div><button className="icon-button" onClick=${onClose}>${icon("close")}</button></div><div className="sheet-body"><${Media} item=${item} detail=${true} /><div className="detail-grid"><div className="detail-panel"><div className="chips" style=${{ marginBottom: "14px" }}><span className="chip">${item.code || "—"}</span><span className="chip">${item.year || "—"}</span><span className="chip">${item.quality || "HD"}</span>${item.episodes_count ? html`<span className="chip">${item.episodes_count} qism</span>` : null}</div><p className="muted" style=${{ margin: 0, lineHeight: 1.7 }}>${item.description || "Tavsif kiritilmagan."}</p>${item.genres?.length ? html`<div className="chips" style=${{ marginTop: "16px" }}>${item.genres.map((genre) => html`<span className="tag" key=${genre}>${genre}</span>`)}</div>` : null}</div><div className="detail-panel"><div className="list"><div className="list-row"><span className="muted">Ko'rishlar</span><strong>${compact(item.views)}</strong></div><div className="list-row"><span className="muted">Yuklab olish</span><strong>${compact(item.downloads)}</strong></div><div className="list-row"><span className="muted">Yoqdi</span><strong>${compact(item.likes)}</strong></div><div className="list-row"><span className="muted">Yoqmadi</span><strong>${compact(item.dislikes)}</strong></div></div><div className="hero-actions"><button className="button" onClick=${() => onBotOpen(item)}>${icon("play")}Davom etish</button><button className="button secondary" onClick=${() => onFavorite(item)}>${item.is_favorite ? "Saqlangan" : "Saqlash"}</button><button className="button ghost" onClick=${() => onShare(item)}>${icon("link")}Ulashish</button></div><div className="content-actions compact"><button className=${joinClass("action-pill", item.user_reaction === "like" && "active")} onClick=${() => onReact(item, "like")} aria-label="Yoqdi">${icon("thumbsUp")}<span>${compact(item.likes)}</span></button><button className=${joinClass("action-pill", item.user_reaction === "dislike" && "active")} onClick=${() => onReact(item, "dislike")} aria-label="Yoqmadi">${icon("thumbsDown")}<span>${compact(item.dislikes)}</span></button></div>${item.episodes?.length ? html`<div style=${{ marginTop: "16px" }}><div className="metric-label">Qismlar</div><div className="chips">${item.episodes.map((episode) => html`<span className="tag" key=${episode.episode_number}>${episode.episode_number}</span>`)}</div></div>` : null}</div></div></div></div></div>`;
}

function ScreenLead({ iconName, title, copy, badge, action }) {
  return html`<section className="section"><div className="screen-lead-card"><div className="screen-lead-main"><div className="screen-lead-icon">${icon(iconName)}</div><div className="screen-lead-copy-block"><div className="eyebrow">Mini app</div><h2 className="screen-lead-title">${title}</h2><p className="screen-lead-copy">${copy}</p></div></div><div className="screen-lead-side">${badge ? html`<span className="screen-lead-badge">${badge}</span>` : null}${action || null}</div></div></section>`;
}

function StatTiles({ items }) {
  return html`<div className="stat-tiles">${items.map((item) => html`<div className="stat-tile" key=${item.label}><div className="stat-tile-icon">${icon(item.iconName)}</div><div className="stat-tile-copy"><span>${item.label}</span><strong>${item.value}</strong></div></div>`)}</div>`;
}

function FeatureRow({ iconName, title, copy, action, badge }) {
  return html`<div className="feature-row"><div className="feature-row-main"><div className="feature-row-icon">${icon(iconName)}</div><div><strong>${title}</strong><p>${copy}</p></div></div><div className="feature-row-side">${badge ? html`<span className="tag">${badge}</span>` : null}${action || null}</div></div>`;
}

function App() {
  const [boot, setBoot] = useState(null);
  const [tab, setTab] = useState("home");
  const [detail, setDetail] = useState(null);
  const [toast, setToast] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [searchSorov, setQidirishSorov] = useState("");
  const [searchType, setQidirishType] = useState("all");
  const [searchResults, setQidirishResults] = useState([]);
  const [searchLoading, setQidirishLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [adForm, setAdForm] = useState({ title: "", description: "", buttonText: "", buttonUrl: "", photoUrl: "" });
  const [uploading, setUploading] = useState(false);
  const [noticeForm, setNoticeForm] = useState({ text: "", link: "" });
  const [proForm, setProForm] = useState({ priceText: "", durationDays: "" });
  const [channelInput, setKanalInput] = useState("");
  const [adminUserSorov, setAdminUserSorov] = useState("");
  const [adminFoydalanuvchi, setAdminFoydalanuvchi] = useState([]);
  const [adminUserLoading, setAdminUserLoading] = useState(false);
  const [adKanalMap, setAdKanalMap] = useState({});
  const [featuredIndex, setFeaturedIndex] = useState(0);

  function notify(message) {
    setToast(message);
    window.clearTimeout(window.__miniappToast);
    window.__miniappToast = window.setTimeout(() => setToast(""), 2600);
  }

  async function loadBoot(silent = false) {
    if (!silent) setLoading(true);
    setError("");
    try {
      const payload = await api("/api/bootstrap");
      setBoot((current) => {
        const prevStamp = current?.notice?.updated_at || "";
        const nextStamp = payload?.notice?.updated_at || "";
        if (silent && nextStamp && prevStamp && nextStamp !== prevStamp) {
          notify("Yangi admin xabari bor");
        }
        return payload;
      });
    } catch (err) {
      setError(err.message || "Yuklashda xato");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadBoot();
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") {
        loadBoot(true);
      }
    }, 30000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    setNoticeForm({
      text: boot?.notice?.text || "",
      link: boot?.notice?.link || "",
    });
  }, [boot?.notice?.updated_at]);

  useEffect(() => {
    setProForm({
      priceText: boot?.settings?.pro_price_text || "",
      durationDays: String(boot?.settings?.pro_duration_days || ""),
    });
  }, [boot?.settings?.pro_price_text, boot?.settings?.pro_duration_days]);

  useEffect(() => {
    if (!adminUserSorov.trim()) {
      setAdminFoydalanuvchi(boot?.admin?.recent_users || []);
    }
  }, [boot?.admin?.recent_users, adminUserSorov]);

  useEffect(() => {
    setMenuOpen(false);
  }, [tab]);

  useEffect(() => {
    const firstKanalId = boot?.admin?.ad_channels?.[0]?.id || "";
    if (!firstKanalId) {
      return;
    }
    setAdKanalMap((current) => {
      const next = { ...current };
      for (const ad of boot?.admin?.pending_ads || []) {
        if (!next[ad.id]) {
          next[ad.id] = firstKanalId;
        }
      }
      return next;
    });
  }, [boot?.admin?.ad_channels, boot?.admin?.pending_ads]);

  async function refreshDetail(item) {
    if (!item) return;
    try {
      const payload = await api(`/api/content/${item.content_type}/${item.id}`);
      setDetail(payload.item);
    } catch (_) {}
  }

  async function openDetail(item) {
    try {
      setBusy(true);
      const payload = await api(`/api/content/${item.content_type}/${item.id}`);
      setDetail(payload.item);
    } catch (err) {
      notify(err.message || "Kontent ochilmadi");
    } finally {
      setBusy(false);
    }
  }

  async function favorite(item) {
    try {
      setBusy(true);
      const payload = await api("/api/favorites/toggle", { method: "POST", body: { contentType: item.content_type, contentRef: item.id } });
      notify(payload.active ? "Saqlangan" : "Saqlangandan olindi");
      await loadBoot(true);
      await refreshDetail(item);
    } catch (err) {
      notify(err.message || "Saqlab bo'lmadi");
    } finally {
      setBusy(false);
    }
  }

  async function react(item, reaction) {
    try {
      setBusy(true);
      await api("/api/reactions", { method: "POST", body: { contentType: item.content_type, contentRef: item.id, reaction } });
      notify("Reaksiya yangilandi");
      await loadBoot(true);
      await refreshDetail(item);
    } catch (err) {
      notify(err.message || "Reaksiya saqlanmadi");
    } finally {
      setBusy(false);
    }
  }

  async function search() {
    if (!searchSorov.trim()) {
      setQidirishResults([]);
      return;
    }
    try {
      setQidirishLoading(true);
      const payload = await api(`/api/search?q=${encodeURIComponent(searchSorov.trim())}&type=${encodeURIComponent(searchType)}`);
      setQidirishResults(payload.items || []);
      notify(`${payload.items?.length || 0} ta natija topildi`);
    } catch (err) {
      notify(err.message || "Qidiruvda xato");
    } finally {
      setQidirishLoading(false);
    }
  }

  async function toggleNotification(key) {
    try {
      await api("/api/notifications/toggle", { method: "POST", body: { key } });
      await loadBoot(true);
      notify("Sozlama yangilandi");
    } catch (err) {
      notify(err.message || "Sozlama saqlanmadi");
    }
  }

  async function uploadRasm(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      setUploading(true);
      const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(new Error("Fayl o'qilmadi"));
        reader.readAsDataURL(file);
      });
      const payload = await api("/api/upload-image", { method: "POST", body: { dataUrl } });
      setAdForm((current) => ({ ...current, photoUrl: payload.url }));
      notify("Rasm yuklandi");
    } catch (err) {
      notify(err.message || "Rasm yuklanmadi");
    } finally {
      setUploading(false);
    }
  }

  async function submitAd(event) {
    event.preventDefault();
    try {
      setBusy(true);
      await api("/api/ads", { method: "POST", body: adForm });
      setAdForm({ title: "", description: "", buttonText: "", buttonUrl: "", photoUrl: "" });
      await loadBoot(true);
      notify("E'lon moderatsiyaga yuborildi");
    } catch (err) {
      notify(err.message || "E'lon yuborilmadi");
    } finally {
      setBusy(false);
    }
  }

  async function setContentMode(mode) {
    try {
      await api("/api/admin/content-mode", { method: "POST", body: { mode } });
      await loadBoot(true);
      notify("Media rejimi yangilandi");
    } catch (err) {
      notify(err.message || "Media rejimi saqlanmadi");
    }
  }

  async function saveNotice(clear = false) {
    try {
      setBusy(true);
      await api("/api/admin/notice", {
        method: "POST",
        body: clear ? { text: "", link: "" } : noticeForm,
      });
      await loadBoot(true);
      notify(clear ? "Sayt xabari tozalandi" : "Sayt xabari saqlandi");
    } catch (err) {
      notify(err.message || "Sayt xabari yangilanmadi");
    } finally {
      setBusy(false);
    }
  }

  async function reviewPayment(requestId, action) {
    try {
      setBusy(true);
      await api("/api/admin/payments/review", {
        method: "POST",
        body: { requestId, action },
      });
      await loadBoot(true);
      notify(action === "approve" ? "To'lov tasdiqlandi" : "To'lov rad etildi");
    } catch (err) {
      notify(err.message || "To'lov ko'rib chiqilmadi");
    } finally {
      setBusy(false);
    }
  }

  async function reviewAd(adId, action) {
    try {
      setBusy(true);
      await api("/api/admin/ads/review", {
        method: "POST",
        body: {
          adId,
          action,
          channelId: action === "approve" ? (adKanalMap[adId] || "") : "",
        },
      });
      await loadBoot(true);
      notify(action === "approve" ? "E'lon joylandi" : "E'lon rad etildi");
    } catch (err) {
      notify(err.message || "E'lon ko'rib chiqilmadi");
    } finally {
      setBusy(false);
    }
  }

  async function saveProSettings() {
    try {
      setBusy(true);
      await api("/api/admin/pro-settings", {
        method: "POST",
        body: {
          priceText: proForm.priceText,
          durationDays: Number(proForm.durationDays),
        },
      });
      await loadBoot(true);
      notify("PRO sozlamalari saqlandi");
    } catch (err) {
      notify(err.message || "PRO sozlamalari saqlanmadi");
    } finally {
      setBusy(false);
    }
  }

  async function createAdKanal() {
    try {
      setBusy(true);
      const payload = await api("/api/admin/ad-channels/create", {
        method: "POST",
        body: { channelRef: channelInput },
      });
      setKanalInput("");
      await loadBoot(true);
      notify(payload.created ? "Kanal qo'shildi" : "Kanal oldin qo'shilgan");
    } catch (err) {
      notify(err.message || "Kanal qo'shilmadi");
    } finally {
      setBusy(false);
    }
  }

  async function deleteAdKanal(channelId) {
    try {
      setBusy(true);
      await api("/api/admin/ad-channels/delete", {
        method: "POST",
        body: { channelId },
      });
      await loadBoot(true);
      notify("Kanal o'chirildi");
    } catch (err) {
      notify(err.message || "Kanal o'chirilmadi");
    } finally {
      setBusy(false);
    }
  }

  async function searchAdminFoydalanuvchi() {
    try {
      setAdminUserLoading(true);
      const payload = await api(`/api/admin/users/search?q=${encodeURIComponent(adminUserSorov.trim())}`);
      setAdminFoydalanuvchi(payload.items || []);
      notify(`${payload.items?.length || 0} ta foydalanuvchi topildi`);
    } catch (err) {
      notify(err.message || "Foydalanuvchi qidirilmadi");
    } finally {
      setAdminUserLoading(false);
    }
  }

  async function setUserPro(userId, enabled) {
    try {
      setBusy(true);
      const payload = await api("/api/admin/users/pro", {
        method: "POST",
        body: { userId, enabled },
      });
      setAdminFoydalanuvchi((current) => current.map((item) => item.id === userId ? payload.item : item));
      await loadBoot(true);
      notify(enabled ? "PRO yoqildi" : "PRO o'chirildi");
    } catch (err) {
      notify(err.message || "PRO holati yangilanmadi");
    } finally {
      setBusy(false);
    }
  }

  async function setUserAdmin(userId, enabled) {
    try {
      setBusy(true);
      const payload = await api("/api/admin/users/admin", {
        method: "POST",
        body: { userId, enabled },
      });
      setAdminFoydalanuvchi((current) => current.map((item) => item.id === userId ? payload.item : item));
      await loadBoot(true);
      notify(enabled ? "Admin huquqi berildi" : "Admin huquqi olindi");
    } catch (err) {
      notify(err.message || "Admin holati yangilanmadi");
    } finally {
      setBusy(false);
    }
  }

  const navItems = useMemo(() => {
    const items = [...NAV];
    if (boot?.user?.is_pro || boot?.user?.is_admin) {
      items.splice(5, 0, { key: "ads", label: "E'lonlar", icon: "megaphone" });
    }
    if (boot?.user?.is_admin) {
      items.push({ key: "admin", label: "Admin", icon: "shield" });
    }
    return items;
  }, [boot?.user?.is_admin, boot?.user?.is_pro]);
  const primaryNavItems = useMemo(() => navItems.filter((item) => PRIMARY_NAV_KEYS.includes(item.key)), [navItems]);
  const secondaryNavItems = useMemo(() => navItems.filter((item) => !PRIMARY_NAV_KEYS.includes(item.key)), [navItems]);
  const secondaryActive = secondaryNavItems.some((item) => item.key === tab);
  const featuredItems = useMemo(() => {
    const pool = [
      ...(boot?.sections?.top_viewed || []),
      ...(boot?.sections?.recent_movies || []),
      ...(boot?.sections?.recent_serials || []),
    ];
    const seen = new Set();
    return pool.filter((item) => {
      const key = `${item.content_type}:${item.id}`;
      if (!item?.id || seen.has(key)) return false;
      seen.add(key);
      return true;
    }).slice(0, 7);
  }, [boot?.sections?.top_viewed, boot?.sections?.recent_movies, boot?.sections?.recent_serials]);
  const newsItems = useMemo(() => {
    const updates = [
      ...(boot?.sections?.recent_movies || []).map((item) => ({ ...item, news_type: "movie" })),
      ...(boot?.sections?.recent_serials || []).map((item) => ({ ...item, news_type: "serial" })),
    ].sort((left, right) => dateValue(right.created_at) - dateValue(left.created_at));
    const rows = [];
    if (boot?.notice?.text) {
      rows.push({
        id: "notice",
        news_type: "notice",
        title: "Admin xabari",
        description: boot.notice.text,
        created_at: boot.notice.updated_at,
        link: boot.notice.link,
      });
    }
    return [...rows, ...updates].slice(0, 12);
  }, [boot?.notice, boot?.sections?.recent_movies, boot?.sections?.recent_serials]);
  const adminQuickActions = useMemo(() => ADMIN_BOT_ACTIONS.slice(0, 6), []);

  useEffect(() => {
    if (!navItems.some((item) => item.key === tab)) {
      setTab("home");
    }
  }, [navItems, tab]);

  useEffect(() => {
    if (featuredItems.length < 2 || tab !== "home") {
      return undefined;
    }
    const timer = window.setInterval(() => {
      setFeaturedIndex((current) => (current + 1) % featuredItems.length);
    }, 5200);
    return () => window.clearInterval(timer);
  }, [featuredItems, tab]);

  if (loading) {
    return html`<div className="loader-wrap"><div className="loader-card"><div className="spinner"></div><div className="eyebrow">Platforma</div><h2 className="headline" style=${{ fontSize: "30px", margin: "8px 0 10px" }}>Yuklanmoqda</h2><p className="subheadline">Ma'lumotlar olinmoqda.</p></div></div>`;
  }

  if (!boot || error) {
    return html`<div className="loader-wrap"><div className="loader-card"><div className="eyebrow">Platforma</div><h2 className="headline" style=${{ fontSize: "30px", margin: "8px 0 10px" }}>Ulanishda xato</h2><p className="subheadline">${error || "Noma'lum xato"}</p><div className="hero-actions"><button className="button" onClick=${() => loadBoot(false)}>Qayta urinish</button></div></div></div>`;
  }

  const sections = boot.sections || {};
  const admin = boot.admin || {};
  const latestItems = [...(sections.recent_movies || []), ...(sections.recent_serials || [])]
    .sort((left, right) => dateValue(right.created_at) - dateValue(left.created_at))
    .slice(0, 8);
  const currentFeatured = featuredItems[featuredIndex % Math.max(featuredItems.length, 1)] || latestItems[0] || null;
  const latestNewsPreview = newsItems.slice(0, 3);
  const featuredMeta = [
    currentFeatured?.year || "—",
    currentFeatured?.quality || "HD",
    `${compact(currentFeatured?.views)} ko'rish`,
    `★ ${Number(currentFeatured?.rating || 0).toFixed(1)}`,
  ];
  const homeShortcuts = [
    { key: "search", iconName: "search", title: "Qidiruv", copy: "Kod va nom" },
    { key: "saved", iconName: "bookmark", title: "Saqlangan", copy: `${sections.favorites?.length || 0} ta` },
    { key: "pro", iconName: "crown", title: "PRO", copy: boot.user.is_pro ? "Faol" : "Tarif" },
    { key: "news", iconName: "spark", title: "Yangiliklar", copy: `${newsItems.length} ta` },
  ];
  const savedStats = [
    { iconName: "bookmark", label: "Saqlangan", value: compact(sections.favorites?.length || 0) },
    { iconName: "stats", label: "Top", value: compact(sections.top_viewed?.length || 0) },
  ];
  const proStats = [
    { iconName: "crown", label: "Tarif", value: boot.settings.pro_price_text || "—" },
    { iconName: "stats", label: "Muddat", value: `${boot.settings.pro_duration_days || 0} kun` },
    { iconName: "link", label: "Kod", value: boot.payment.code || "—" },
  ];
  const proFeatures = [
    { iconName: "spark", title: "Yangi kontent", copy: "Premium foydalanuvchilar uchun tezkor xabarlar." },
    { iconName: "megaphone", title: "E'lon joylash", copy: "Moderatsiyadan o'tgan postlarni joylash imkoniyati." },
    { iconName: "bookmark", title: "Sinxron tarix", copy: "Saqlanganlar bot va ilova o'rtasida bir xil qoladi." },
    { iconName: "shield", title: "Yagona tarif", copy: "Narx va muddat admin tomonidan markaziy boshqariladi." },
  ];
  const profileStats = [
    { iconName: "bookmark", label: "Saqlangan", value: compact(sections.favorites?.length || 0) },
    { iconName: "stats", label: "Top kartalar", value: compact(sections.top_viewed?.length || 0) },
    { iconName: "crown", label: "Status", value: boot.user.is_pro ? "PRO" : "Oddiy" },
  ];
  const adminOverview = [
    { iconName: "user", label: "Foydalanuvchilar", value: compact(admin.total_users || 0) },
    { iconName: "crown", label: "PRO", value: compact(admin.total_pro_users || 0) },
    { iconName: "play", label: "Kinolar", value: compact(admin.total_movies || 0) },
    { iconName: "grid", label: "Seriallar", value: compact(admin.total_serials || 0) },
  ];
  const shellClass = joinClass("app-shell", tab === "admin" && boot.user.is_admin && "app-shell-admin");

  return html`
      <div className=${shellClass}>
        ${toast ? html`<div className="toast">${toast}</div>` : null}
       ${boot.notice?.text ? html`<section className="notice-bar"><div className="notice-copy"><div className="eyebrow">Yangilik</div><strong>${boot.notice.text}</strong><span className="muted">${dateText(boot.notice.updated_at)}</span></div><div className="notice-actions">${boot.notice.link ? html`<button className="button secondary" onClick=${() => openUrl(boot.notice.link)}>Batafsil</button>` : null}<button className="button ghost" onClick=${() => setTab("news")}>Yangiliklar</button></div></section>` : null}
       <header className="topbar">
         <div className="brand brand-row">
           <button className="icon-button topbar-icon menu-trigger" onClick=${() => setMenuOpen(true)} aria-label="Menyu">
             ${icon("menu")}
           </button>
           <div className="logo-mark"></div>
           <div className="brand-meta">
             <div className="eyebrow">Telegram platforma</div>
             <h1 className="headline accent-headline">Mir Top Kino</h1>
             <p className="subheadline">Premium kino va serial katalogi.</p>
           </div>
         </div>
         <div className="header-actions">
           <div className="header-badges">
             ${boot.user.is_pro ? html`<span className="pill badge-pill pro">${icon("crown")}PRO faol</span>` : null}
             ${boot.user.is_admin ? html`<span className="pill badge-pill admin">${icon("shield")}ADMIN</span>` : null}
           </div>
           <button className="avatar-button" onClick=${() => setTab("profile")} aria-label="Profil">
             ${initialsFromName(boot.user.full_name || boot.user.username || boot.user.id)}
           </button>
         </div>
      </header>

       ${tab === "home" ? html`
         <section className="hero hero-premium hero-cinema">
           <div className="panel hero-main premium-hero featured-stage" style=${currentFeatured?.preview_url ? { backgroundImage: `linear-gradient(180deg, rgba(5, 9, 18, 0.16), rgba(5, 9, 18, 0.92)), url(${currentFeatured.preview_url})` } : {}}>
             <div className="featured-stage-inner">
               <div className="featured-copy-stack">
                 <div className="eyebrow">Tanlangan premyera</div>
                 <div className="hero-badge-row">
                   <span className="hero-badge">${currentFeatured ? (currentFeatured.content_type === "serial" ? "Serial" : "Kino") : "Tanlangan"}</span>
                   <span className="hero-badge">${currentFeatured?.code || "Kod tayinlanmagan"}</span>
                 </div>
                 <p className="featured-lead">Telegram ichida eng yangi kino va seriallar.</p>
                 <h2 className="featured-title">${currentFeatured?.title || "Kontent tayyorlanmoqda"}</h2>
                 <div className="featured-inline-meta">
                   ${featuredMeta.map((item) => html`<span key=${item}>${item}</span>`)}
                 </div>
                 <p className="featured-copy">${currentFeatured?.description || "Yangi qo'shilgan kontent va eng ko'p ochilgan kartalar shu yerda chiqadi."}</p>
                 <div className="hero-actions">
                   <button className="button hero-primary" onClick=${() => currentFeatured ? openDetail(currentFeatured) : setTab("search")} disabled=${busy}>${icon("play")}Davom etish</button>
                   <button className="button secondary hero-secondary" onClick=${() => currentFeatured ? favorite(currentFeatured) : setTab("saved")} disabled=${busy}>${icon("bookmark")}${currentFeatured?.is_favorite ? "Saqlangan" : "Saqlash"}</button>
                   <button className="icon-button hero-icon-button" onClick=${() => currentFeatured ? openInBot(currentFeatured, notify, boot?.links) : setTab("search")} aria-label="Botda ochish">${icon("arrowRight")}</button>
                 </div>
                 ${featuredItems.length > 1 ? html`<div className="hero-dots">${featuredItems.map((item, index) => html`<button className=${joinClass("hero-dot", index === (featuredIndex % Math.max(featuredItems.length, 1)) && "active")} key=${`${item.content_type}-${item.id}`} onClick=${() => setFeaturedIndex(index)} aria-label=${item.title}></button>`)}</div>` : null}
               </div>
             </div>
           </div>
         </section>

         <section className="section">
           <div className="shortcut-grid">
             ${homeShortcuts.map((item) => html`<button className="shortcut-card" key=${item.key} onClick=${() => setTab(item.key)}><div className="shortcut-icon">${icon(item.iconName)}</div><div><strong>${item.title}</strong><span>${item.copy}</span></div>${icon("arrowRight")}</button>`)}
           </div>
         </section>

        <section className="section">
          <${SectionSarlavha}
            iconName="spark"
            title="Yangiliklar"
            copy="Yangi qo'shilgan kontent va admin xabarlari"
            action=${html`<button className="button secondary" onClick=${() => setTab("news")}>Barchasi</button>`}
          />
          ${latestNewsPreview.length ? html`<div className="news-grid">${latestNewsPreview.map((item) => html`<button className="news-card" key=${item.id || item.created_at} onClick=${() => item.news_type === "notice" ? (item.link ? openUrl(item.link) : setTab("news")) : openDetail(item)}><div className="news-card-head"><span className="tag">${item.news_type === "notice" ? "Xabar" : item.news_type === "serial" ? "Serial" : "Kino"}</span><span className="muted">${dateText(item.created_at)}</span></div><strong>${item.title || "Yangilik"}</strong><p>${item.description || "Yangi kontent qo'shildi."}</p></button>`)}</div>` : html`<${Empty} title="Yangilik yo'q" copy="Yangi xabar yoki kontent chiqqanda shu yerda ko'rinadi." />`}
        </section>

        <${RailSection}
          iconName="stats"
          title="Trenddagi kontent"
          copy="Eng ko'p ochilgan kontentlar"
          items=${sections.top_viewed || []}
          onOpen=${openDetail}
          onFavorite=${favorite}
          onReact=${react}
        />

        <${RailSection}
          iconName="play"
          title="Yangi kinolar"
          copy="Oxirgi qo'shilgan kinolar"
          items=${sections.recent_movies || []}
          onOpen=${openDetail}
          onFavorite=${favorite}
          onReact=${react}
        />

        <${RailSection}
          iconName="grid"
          title="Yangi seriallar"
          copy="Oxirgi qo'shilgan seriallar"
          items=${sections.recent_serials || []}
          onOpen=${openDetail}
          onFavorite=${favorite}
          onReact=${react}
        />

        <${RailSection}
          iconName="bookmark"
          title="Yangi katalog"
          copy="Kino va seriallardan aralash yangi kartalar"
          items=${latestItems}
          onOpen=${openDetail}
          onFavorite=${favorite}
          onReact=${react}
        />

        ${boot.user.is_admin ? html`
          <section className="section">
            <${SectionSarlavha} iconName="shield" title="Tezkor admin amallari" copy="Botdagi asosiy bo'limlarga bir tegishda o'tish" />
            <div className="quick-grid">
              ${adminQuickActions.map((item) => html`<button className="quick-action-card" key=${item.action} onClick=${() => sendToBot({ action: item.action }, notify, boot?.links)}>${icon(item.icon)}<div><strong>${item.label}</strong><span>Botga o'tish</span></div>${icon("arrowRight")}</button>`)}
            </div>
          </section>
        ` : null}
      ` : null}

      ${tab === "search" ? html`
        <${ScreenLead} iconName="search" title="Qidiruv" copy="Kod, nom yoki tavsif bo'yicha qidiring." badge=${searchSorov.trim() ? `${searchResults.length} natija` : "Jonli qidiruv"} />
        <section className="section">
          <div className="panel search-box">
            <div className="field"><label>So'rov</label><input className="input" value=${searchSorov} onInput=${(event) => setQidirishSorov(event.target.value)} placeholder="Kino kodi yoki nomi" /></div>
            <div className="chips">${SEARCH_TYPES.map((item) => html`<button className=${joinClass("chip", searchType === item.key && "active")} key=${item.key} onClick=${() => setQidirishType(item.key)}>${item.label}</button>`)}</div>
            <div className="hero-actions"><button className="button" onClick=${search} disabled=${searchLoading}>${searchLoading ? "Qidirilmoqda..." : "Qidirish"}</button></div>
          </div>
        </section>
        <section className="section">
          ${searchLoading ? html`<${SkeletonGrid} count=${6} />` : searchResults?.length ? html`<div className="content-grid">${searchResults.map((item) => html`<${Card} key=${item.id} item=${item} onOpen=${openDetail} onFavorite=${favorite} onReact=${react} />`)}</div>` : html`<${Empty} title="Natija topilmadi" copy="Kod yoki nom yozib qidirishni boshlang." />`}
        </section>
      ` : null}

      ${tab === "saved" ? html`
        <${ScreenLead} iconName="bookmark" title="Saqlanganlar" copy="Bot va mini app o'rtasida bir xil turadigan sevimli kontentlar." badge=${`${sections.favorites?.length || 0} ta`} />
        ${sections.favorites?.length ? html`<section className="section"><${StatTiles} items=${savedStats} /></section>` : null}
        <section className="section">
          ${sections.favorites?.length ? html`<div className="content-grid">${sections.favorites.map((item) => html`<${Card} key=${item.id} item=${item} onOpen=${openDetail} onFavorite=${favorite} onReact=${react} />`)}</div>` : html`<${Empty} title="Saqlanganlar bo'sh" copy="Istalgan kartadagi saqlash tugmasi orqali bu yerga qo'shing." />`}
        </section>
      ` : null}

      ${tab === "pro" ? html`
        <${ScreenLead} iconName="crown" title="PRO bo'limi" copy="Yagona tarif, premium kirish va to'lov havolalari shu yerda." badge=${boot.user.is_pro ? "PRO faol" : "Oddiy holat"} />
        <section className="section"><${StatTiles} items=${proStats} /></section>
        <section className="section">
          <div className="hero">
            <div className="panel hero-main">
              <div className="eyebrow">Joriy holat</div>
              <h2 style=${{ margin: "8px 0 10px", fontSize: "32px" }}>${boot.user.is_pro ? "PRO faol" : "PRO kerak"}</h2>
              <p className="subheadline">Narx: ${boot.settings.pro_price_text}<br />Muddat: ${boot.settings.pro_duration_days} kun<br />Qachongacha: ${boot.user.pro_until || "—"}</p>
              <div className="hero-actions">
                <button className="button" onClick=${() => sendToBot({ action: "open_pro" }, notify, boot?.links)}>Davom etish</button>
                <button className="button secondary" onClick=${() => copyText(boot.payment.code, notify)}>Koddan nusxa olish</button>
              </div>
            </div>
            <div className="panel hero-side">
              <div className="list">
                <div className="list-card"><div className="metric-label">To'lov kodi</div><div style=${{ fontWeight: 800, fontSize: "24px" }}>${boot.payment.code}</div></div>
                <div className="list-card"><div className="metric-label">Telegram ID</div><div style=${{ fontWeight: 800, fontSize: "24px" }}>${boot.user.id}</div></div>
              </div>
              <div className="hero-actions">${(boot.payment.links || []).map((link, index) => html`<button className="button secondary" key=${link} onClick=${() => openUrl(link)}>${icon("link")}To'lov ${index + 1}</button>`)}</div>
            </div>
          </div>
        </section>
        <section className="section"><div className="feature-list">${proFeatures.map((item) => html`<${FeatureRow} key=${item.title} iconName=${item.iconName} title=${item.title} copy=${item.copy} />`)}</div></section>
      ` : null}

      ${tab === "news" ? html`
        <${ScreenLead} iconName="spark" title="Yangiliklar" copy="Admin e'lonlari va oxirgi qo'shilgan kontentlar shu bo'limda." badge=${`${newsItems.length} ta`} />
        <section className="section">
          ${newsItems.length ? html`<div className="news-feed">${newsItems.map((item) => html`<article className="news-feed-item" key=${item.id || item.created_at}>${item.news_type !== "notice" ? html`<div className="news-feed-media">${item.preview_url ? html`<img src=${item.preview_url} alt=${item.title || "Yangilik"} loading="lazy" />` : html`<div className="news-feed-fallback">${icon(item.news_type === "serial" ? "grid" : "play")}</div>`}</div>` : html`<div className="news-feed-media notice">${icon("spark")}</div>`}<div className="news-feed-content"><div className="news-feed-top"><div><div className="eyebrow">${item.news_type === "notice" ? "Admin xabari" : item.news_type === "serial" ? "Yangi serial" : "Yangi kino"}</div><h3>${item.title || "Yangilik"}</h3></div><span className="tag">${dateText(item.created_at)}</span></div><p>${item.description || "Yangi kontent qo'shildi."}</p><div className="hero-actions">${item.news_type === "notice" ? html`${item.link ? html`<button className="button secondary" onClick=${() => openUrl(item.link)}>Batafsil</button>` : null}` : html`<button className="button" onClick=${() => openDetail(item)}>Ko'rish</button><button className="button ghost" onClick=${() => openInBot(item, notify, boot?.links)}>Botda davom etish</button>`}</div></div></article>`)}</div>` : html`<${Empty} title="Yangilik yo'q" copy="Yangi xabarlar va kontent shu bo'limda chiqadi." />`}
        </section>
      ` : null}

      ${tab === "ads" ? html`
        <section className="section">
          <${SectionSarlavha} iconName="grid" title="Mening e'lonlarim" copy="Oxirgi e'lonlar va ularning holati" />
          ${boot.ads?.mine?.length ? html`<div className="list">${boot.ads.mine.map((item) => html`<div className="list-card" key=${item.id}><div className="list-row"><div><div style=${{ fontWeight: 700 }}>${item.title}</div><div className="muted">${statusText(item.status)} · ${dateText(item.created_at)}</div></div><div className="tag">${statusText(item.status)}</div></div><p className="muted" style=${{ marginBottom: 0 }}>${item.description}</p></div>`)}</div>` : html`<${Empty} title="E'lonlar yo'q" copy="Yuborgan e'lonlaringiz shu yerda chiqadi." />`}
        </section>

        <section className="section">
          <${SectionSarlavha} iconName="megaphone" title="E'lon yaratish" copy="Faqat aktiv PRO foydalanuvchilar uchun" />
          ${boot.ads?.can_create ? html`<div className="hero"><form className="panel search-box" onSubmit=${submitAd}><div className="form-grid"><div className="field"><label>Sarlavha</label><input className="input" value=${adForm.title} onInput=${(event) => setAdForm((current) => ({ ...current, title: event.target.value }))} placeholder="E'lon sarlavhasi" /></div><div className="field"><label>Tugma matni</label><input className="input" value=${adForm.buttonText} onInput=${(event) => setAdForm((current) => ({ ...current, buttonText: event.target.value }))} placeholder="Ixtiyoriy" /></div></div><div className="field"><label>Tavsif</label><textarea className="textarea" value=${adForm.description} onInput=${(event) => setAdForm((current) => ({ ...current, description: event.target.value }))} placeholder="Tavsif"></textarea></div><div className="field"><label>Tugma havolasi</label><input className="input" value=${adForm.buttonUrl} onInput=${(event) => setAdForm((current) => ({ ...current, buttonUrl: event.target.value }))} placeholder="https://..." /></div><div className="hero-actions"><button className="button" type="submit" disabled=${busy}>Moderatsiyaga yuborish</button><button className="button ghost" type="button" onClick=${() => setAdForm({ title: "", description: "", buttonText: "", buttonUrl: "", photoUrl: "" })}>Tozalash</button></div></form><div className="panel hero-side"><div className="upload-box"><div className="metric-label">Rasm</div>${adForm.photoUrl ? html`<img className="upload-preview" src=${adForm.photoUrl} alt="e'lon rasmi" />` : html`<div className="upload-preview" style=${{ display: "grid", placeItems: "center" }}>Rasm preview</div>`}<input className="input" type="file" accept="image/*" onChange=${uploadRasm} /><div className="muted">${uploading ? "Yuklanmoqda..." : "E'lon uchun ixtiyoriy rasm yuklang."}</div></div></div></div>` : html`<div className="panel hero-main"><div className="eyebrow">Ruxsat yopiq</div><h2 style=${{ margin: "8px 0 10px", fontSize: "32px" }}>PRO kerak</h2><p className="subheadline">E'lon joylash uchun PRO faolligi kerak.</p><div className="hero-actions"><button className="button" onClick=${() => sendToBot({ action: "open_pro" }, notify, boot?.links)}>PRO olish</button></div></div>`}
        </section>
      ` : null}

      ${tab === "profile" ? html`
        <${ScreenLead} iconName="user" title="Profil" copy="Hisob, ID va bildirishnoma sozlamalari shu bo'limda." badge=${boot.user.is_admin ? "Admin" : boot.user.is_pro ? "PRO" : "Oddiy"} />
        <section className="section">
          <div className="profile-hero-card"><div className="profile-avatar">${initialsFromName(boot.user.full_name || boot.user.username || boot.user.id)}</div><div className="profile-hero-copy"><h3>${boot.user.full_name || boot.user.username || boot.user.id}</h3><p>Telegram ID: ${boot.user.id}</p><div className="profile-hero-badges"><span className="tag">${boot.user.is_pro ? "PRO faol" : "Oddiy foydalanuvchi"}</span>${boot.user.is_admin ? html`<span className="tag">Admin</span>` : null}</div></div></div>
        </section>
        <section className="section"><${StatTiles} items=${profileStats} /></section>
        <section className="section">
          <div className="feature-list">
            <${FeatureRow} iconName="link" title="Telegram ID" copy=${`ID: ${boot.user.id}`} action=${html`<button className="button secondary" onClick=${() => copyText(boot.user.id, notify)}>Nusxa</button>`} />
            <${FeatureRow} iconName="search" title="Bot sozlamalari" copy="Telegram bot ichidagi qo'shimcha boshqaruv bo'limlari." action=${html`<button className="button ghost" onClick=${() => sendToBot({ action: "open_notifications" }, notify, boot?.links)}>Botga o'tish</button>`} />
            <${FeatureRow} iconName="spark" title="Yangi kontent" copy="Yangi kino va seriallar haqida xabar." action=${html`<button className=${joinClass("button", boot.notifications.new_content ? "success" : "ghost")} onClick=${() => toggleNotification("new_content")}>${boot.notifications.new_content ? "Yoqilgan" : "O'chiq"}</button>`} />
            <${FeatureRow} iconName="crown" title="PRO yangiliklari" copy="PRO status va tarif yangilanishlari." action=${html`<button className=${joinClass("button", boot.notifications.pro_updates ? "success" : "ghost")} onClick=${() => toggleNotification("pro_updates")}>${boot.notifications.pro_updates ? "Yoqilgan" : "O'chiq"}</button>`} />
            <${FeatureRow} iconName="megaphone" title="E'lon xabarlari" copy="E'lon moderatsiyasi va status xabarlari." action=${html`<button className=${joinClass("button", boot.notifications.ads_updates ? "success" : "ghost")} onClick=${() => toggleNotification("ads_updates")}>${boot.notifications.ads_updates ? "Yoqilgan" : "O'chiq"}</button>`} />
          </div>
        </section>
      ` : null}

      ${tab === "admin" && boot.user.is_admin ? html`
        <${ScreenLead} iconName="shield" title="Admin panel" copy="Foydalanuvchilar, to'lovlar, e'lonlar va global sozlamalar shu yerda boshqariladi." badge="To'liq kirish" />
        <section className="section"><${StatTiles} items=${adminOverview} /></section>
        <section className="section">
          <${SectionSarlavha} iconName="shield" title="Tezkor amallar" copy="Botdagi asosiy oqimlarga bir tegishda o'tish" />
          <div className="quick-grid">
            ${ADMIN_BOT_ACTIONS.map((item) => html`<button className="quick-action-card" key=${item.action} onClick=${() => sendToBot({ action: item.action }, notify, boot?.links)}>${icon(item.icon)}<div><strong>${item.label}</strong><span>Botga o'tish</span></div>${icon("arrowRight")}</button>`)}
          </div>
        </section>
        <section className="section">
          <${SectionSarlavha} iconName="shield" title="Global media rejimi" copy="Barcha foydalanuvchilarga bir xil qo'llanadi." />
          <div className="hero"><div className="panel hero-main"><div className="eyebrow">Joriy holat</div><h2 style=${{ margin: "8px 0 10px", fontSize: "32px" }}>${boot.settings.content_mode_label}</h2><p className="subheadline">Video va media himoyasi uchun global rejim shu yerda boshqariladi.</p><div className="hero-actions"><button className="button secondary" onClick=${() => setContentMode("private")}>Yopiq</button><button className="button secondary" onClick=${() => setContentMode("public")}>Ochiq</button></div></div></div>
        </section>
        <section className="section">
          <${SectionSarlavha} iconName="user" title="Foydalanuvchilar" copy="ID yoki ism bo'yicha toping va huquqlarni boshqaring." />
          <div className="panel search-box">
            <div className="form-grid">
              <div className="field"><label>Qidiruv</label><input className="input" value=${adminUserSorov} onInput=${(event) => setAdminUserSorov(event.target.value)} placeholder="Telegram ID yoki ism" /></div>
              <div className="hero-actions"><button className="button" onClick=${searchAdminFoydalanuvchi} disabled=${adminUserLoading}>${adminUserLoading ? "Qidirilmoqda..." : "Qidirish"}</button><button className="button ghost" onClick=${() => { setAdminUserSorov(""); setAdminFoydalanuvchi(admin.recent_users || []); }} disabled=${adminUserLoading}>So'nggilari</button></div>
            </div>
          </div>
          ${(adminFoydalanuvchi || []).length ? html`<div className="list" style=${{ marginTop: "14px" }}>${(adminFoydalanuvchi || []).map((item) => html`<div className="list-card" key=${item.id}><div className="list-row"><div><div style=${{ fontWeight: 700 }}>${item.full_name || `User ${item.id}`}</div><div className="muted">ID ${item.id} · Qo'shilgan: ${dateText(item.joined_at)}</div></div><div className="chips"><span className="tag">${item.is_pro ? "PRO" : "Oddiy"}</span><span className="tag">${item.is_admin ? "Admin" : "Foydalanuvchi"}</span></div></div><div className="muted" style=${{ marginTop: "10px" }}>PRO muddati: ${item.pro_until || "—"}</div>${item.is_seed_admin ? html`<div className="muted" style=${{ marginTop: "6px" }}>Asosiy admin: env orqali boshqariladi</div>` : null}<div className="hero-actions"><button className=${joinClass("button", item.is_pro ? "danger" : "success")} onClick=${() => setUserPro(item.id, !item.is_pro)} disabled=${busy}>${item.is_pro ? "PRO o'chirish" : "PRO yoqish"}</button><button className=${joinClass("button", item.is_admin ? "danger" : "secondary")} onClick=${() => setUserAdmin(item.id, !item.is_admin)} disabled=${busy || (item.id === boot.user.id && item.is_admin) || item.is_seed_admin}>${item.is_admin ? "Admin olish" : "Admin berish"}</button></div></div>`)}</div>` : html`<div style=${{ marginTop: "14px" }}><${Empty} title="Foydalanuvchi topilmadi" copy="Telegram ID yoki ism orqali qidiring." /></div>`}
        </section>
        <section className="section">
          <${SectionSarlavha} iconName="grid" title="Sayt xabari" copy="Bu xabar ilova ichida barcha foydalanuvchilarga ko'rinadi." />
          <div className="panel search-box">
            <div className="field"><label>Xabar matni</label><textarea className="textarea" value=${noticeForm.text} onInput=${(event) => setNoticeForm((current) => ({ ...current, text: event.target.value }))} placeholder="Qisqa e'lon yoki bildirishnoma"></textarea></div>
            <div className="field"><label>Havola</label><input className="input" value=${noticeForm.link} onInput=${(event) => setNoticeForm((current) => ({ ...current, link: event.target.value }))} placeholder="Ixtiyoriy https://... yoki /app/" /></div>
            <div className="hero-actions"><button className="button" onClick=${() => saveNotice(false)} disabled=${busy}>Saqlash</button><button className="button ghost" onClick=${() => saveNotice(true)} disabled=${busy}>Tozalash</button></div>
          </div>
        </section>
        <section className="section">
          <${SectionSarlavha} iconName="crown" title="PRO sozlamalari" copy="Bot va ilova uchun yagona tarif" />
          <div className="panel search-box">
            <div className="form-grid">
              <div className="field"><label>Narx matni</label><input className="input" value=${proForm.priceText} onInput=${(event) => setProForm((current) => ({ ...current, priceText: event.target.value }))} placeholder="12000 so'm" /></div>
              <div className="field"><label>Muddat (kun)</label><input className="input" type="number" min="1" max="3650" value=${proForm.durationDays} onInput=${(event) => setProForm((current) => ({ ...current, durationDays: event.target.value }))} placeholder="30" /></div>
            </div>
            <div className="hero-actions"><button className="button" onClick=${saveProSettings} disabled=${busy}>Saqlash</button></div>
          </div>
        </section>
        <section className="section">
          <${SectionSarlavha} iconName="megaphone" title="E'lon kanallari" copy="Tasdiqlangan e'lonlar shu kanallarga joylanadi" />
          <div className="hero">
            <div className="panel search-box">
              <div className="field"><label>Kanal</label><input className="input" value=${channelInput} onInput=${(event) => setKanalInput(event.target.value)} placeholder="@kanal yoki -100..." /></div>
              <div className="hero-actions"><button className="button" onClick=${createAdKanal} disabled=${busy || !channelInput.trim()}>Kanal qo'shish</button></div>
            </div>
            <div className="panel hero-side">
              ${(admin.ad_channels || []).length ? html`<div className="list">${(admin.ad_channels || []).map((channel) => html`<div className="list-card" key=${channel.id}><div className="list-row"><div><div style=${{ fontWeight: 700 }}>${channel.title}</div><div className="muted">${channel.channel_ref}</div></div><button className="button danger" onClick=${() => deleteAdKanal(channel.id)} disabled=${busy}>O'chirish</button></div></div>`)}</div>` : html`<${Empty} title="Kanallar yo'q" copy="Tasdiqlash uchun kamida bitta kanal qo'shing." />`}
            </div>
          </div>
        </section>
        <section className="section"><${SectionSarlavha} iconName="stats" title="Kutilayotgan PRO to'lovlar" copy="Tasdiqlanishi kerak bo'lgan so'rovlar" />${admin.pending_payments?.length ? html`<div className="list">${admin.pending_payments.map((item) => html`<div className="list-card" key=${item.id}><div className="list-row"><div><div style=${{ fontWeight: 700 }}>Foydalanuvchi ${item.user_tg_id}</div><div className="muted">${item.payment_code || "Kod yo'q"} · ${dateText(item.created_at)}</div></div><div className="tag">${statusText(item.status)}</div></div>${item.comment ? html`<p className="muted">${item.comment}</p>` : null}<div className="hero-actions"><button className="button success" onClick=${() => reviewPayment(item.id, "approve")} disabled=${busy}>Tasdiqlash</button><button className="button danger" onClick=${() => reviewPayment(item.id, "reject")} disabled=${busy}>Rad etish</button></div></div>`)}</div>` : html`<${Empty} title="Kutilayotgan to'lov yo'q" copy="Barcha PRO so'rovlar ko'rib chiqilgan." />`}</section>
        <section className="section"><${SectionSarlavha} iconName="megaphone" title="Kutilayotgan e'lonlar" copy="Moderatsiya navbatidagi e'lonlar" />${admin.pending_ads?.length ? html`<div className="list">${admin.pending_ads.map((item) => html`<div className="list-card" key=${item.id}><div className="list-row"><div><div style=${{ fontWeight: 700 }}>${item.title}</div><div className="muted">Foydalanuvchi ${item.user_tg_id} · ${dateText(item.created_at)}</div></div><div className="tag">${statusText(item.status)}</div></div><p className="muted">${item.description}</p><div className="field"><label>Kanal</label><select className="input" value=${adKanalMap[item.id] || ""} onChange=${(event) => setAdKanalMap((current) => ({ ...current, [item.id]: event.target.value }))}>${(admin.ad_channels || []).map((channel) => html`<option value=${channel.id} key=${channel.id}>${channel.title}</option>`)}</select></div><div className="hero-actions"><button className="button success" onClick=${() => reviewAd(item.id, "approve")} disabled=${busy || !(admin.ad_channels || []).length}>Tasdiqlash</button><button className="button danger" onClick=${() => reviewAd(item.id, "reject")} disabled=${busy}>Rad etish</button></div></div>`)}</div>` : html`<${Empty} title="Kutilayotgan e'lon yo'q" copy="Moderatsiya navbati bo'sh." />`}</section>
      ` : null}

      <nav className="bottom-nav">
        <div className="bottom-nav-inner">
          ${primaryNavItems.map((item) => html`<button className=${joinClass("nav-item", tab === item.key && "active")} key=${item.key} onClick=${() => setTab(item.key)}>${icon(item.icon)}<span className="nav-label">${item.label}</span></button>`)}
          <button className=${joinClass("nav-item", (menuOpen || secondaryActive) && "active")} onClick=${() => setMenuOpen((current) => !current)}>${icon("menu")}<span className="nav-label">Menyu</span></button>
        </div>
      </nav>
      ${menuOpen ? html`<div className="menu-backdrop" onClick=${() => setMenuOpen(false)}><div className="menu-sheet" onClick=${(event) => event.stopPropagation()}><div className="menu-sheet-header"><div><div className="eyebrow">Bo'limlar</div><h3 style=${{ margin: "8px 0 0", fontSize: "24px" }}>Menyu</h3></div><button className="icon-button" onClick=${() => setMenuOpen(false)}>${icon("close")}</button></div><div className="menu-grid">${secondaryNavItems.map((item) => html`<button className=${joinClass("menu-card", tab === item.key && "active")} key=${item.key} onClick=${() => setTab(item.key)}><div className="menu-card-icon">${icon(item.icon)}</div><div className="menu-card-label">${item.label}</div></button>`)}</div></div></div>` : null}
      <${DetailSheet} item=${detail} onClose=${() => setDetail(null)} onFavorite=${favorite} onReact=${react} onBotOpen=${(item) => openInBot(item, notify, boot?.links)} onShare=${(item) => shareItem(item, notify, boot?.links)} />
    </div>
  `;
}

createRoot(document.getElementById("root")).render(html`<${App} />`);
