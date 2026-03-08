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
  { key: "home", label: "Home", icon: "home" },
  { key: "search", label: "Search", icon: "search" },
  { key: "saved", label: "Saved", icon: "bookmark" },
  { key: "pro", label: "Pro", icon: "crown" },
  { key: "ads", label: "Ads", icon: "megaphone" },
  { key: "profile", label: "Profile", icon: "user" },
];

const PRIMARY_NAV_KEYS = ["home", "search", "saved", "pro"];

const SEARCH_TYPES = [
  { key: "all", label: "All" },
  { key: "movie", label: "Movies" },
  { key: "serial", label: "Serials" },
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
  const payload = await response.json().catch(() => ({ ok: false, detail: "Invalid response" }));
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.detail || "Request failed");
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

function copyText(value, notify) {
  navigator.clipboard.writeText(String(value || "")).then(() => notify("Copied")).catch(() => notify("Copy failed"));
}

function sendToBot(payload, notify) {
  if (tg?.sendData) {
    tg.sendData(JSON.stringify(payload));
    tg.close();
    return;
  }
  notify("Open this app from Telegram");
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

function SectionTitle({ iconName, title, copy, action }) {
  return html`<div className="section-header"><div><h2 className="section-title">${icon(iconName)}${title}</h2>${copy ? html`<p className="section-copy">${copy}</p>` : null}</div>${action || null}</div>`;
}

function Card({ item, onOpen, onFavorite, onReact }) {
  return html`<article className="content-card" onClick=${() => onOpen(item)}><${Media} item=${item} /><div className="content-card-body"><div className="content-card-top"><div><h3 className="content-title">${item.title || "Untitled"}</h3><div className="content-meta"><span>${item.code || "—"}</span><span>${item.year || "—"}</span><span>${item.quality || "HD"}</span></div></div><div className="tag">${item.content_type === "serial" ? "Serial" : "Movie"}</div></div><div className="content-meta"><span>${compact(item.views)} views</span><span>${compact(item.likes)} likes</span><span>${Number(item.rating || 0).toFixed(1)} rating</span></div><div className="content-actions" onClick=${(event) => event.stopPropagation()}><button className=${joinClass("icon-button", item.is_favorite && "active")} onClick=${() => onFavorite(item)}>${icon("bookmark")}</button><button className=${joinClass("icon-button", item.user_reaction === "like" && "active")} onClick=${() => onReact(item, "like")}>Like</button><button className=${joinClass("icon-button", item.user_reaction === "dislike" && "active")} onClick=${() => onReact(item, "dislike")}>Dislike</button><button className="button secondary" onClick=${() => onOpen(item)}>Open</button></div></div></article>`;
}

function DetailSheet({ item, onClose, onFavorite, onReact, onBotOpen }) {
  if (!item) return null;
  return html`<div className="sheet-backdrop" onClick=${onClose}><div className="sheet" onClick=${(event) => event.stopPropagation()}><div className="sheet-header"><div><div className="eyebrow">${item.content_type === "serial" ? "Serial detail" : "Movie detail"}</div><h2 style=${{ margin: "8px 0 0", fontSize: "28px" }}>${item.title}</h2></div><button className="icon-button" onClick=${onClose}>${icon("close")}</button></div><div className="sheet-body"><${Media} item=${item} detail=${true} /><div className="detail-grid"><div className="detail-panel"><div className="chips" style=${{ marginBottom: "14px" }}><span className="chip">${item.code || "—"}</span><span className="chip">${item.year || "—"}</span><span className="chip">${item.quality || "HD"}</span>${item.episodes_count ? html`<span className="chip">${item.episodes_count} episodes</span>` : null}</div><p className="muted" style=${{ margin: 0, lineHeight: 1.7 }}>${item.description || "No description."}</p>${item.genres?.length ? html`<div className="chips" style=${{ marginTop: "16px" }}>${item.genres.map((genre) => html`<span className="tag" key=${genre}>${genre}</span>`)}</div>` : null}</div><div className="detail-panel"><div className="list"><div className="list-row"><span className="muted">Views</span><strong>${compact(item.views)}</strong></div><div className="list-row"><span className="muted">Downloads</span><strong>${compact(item.downloads)}</strong></div><div className="list-row"><span className="muted">Likes</span><strong>${compact(item.likes)}</strong></div><div className="list-row"><span className="muted">Dislikes</span><strong>${compact(item.dislikes)}</strong></div></div><div className="hero-actions"><button className="button" onClick=${() => onBotOpen(item)}>${icon("play")}Open in bot</button><button className="button secondary" onClick=${() => onFavorite(item)}>${item.is_favorite ? "Saved" : "Save"}</button></div><div className="hero-actions"><button className="button ghost" onClick=${() => onReact(item, "like")}>Like</button><button className="button ghost" onClick=${() => onReact(item, "dislike")}>Dislike</button></div>${item.episodes?.length ? html`<div style=${{ marginTop: "16px" }}><div className="metric-label">Episodes</div><div className="chips">${item.episodes.map((episode) => html`<span className="tag" key=${episode.episode_number}>${episode.episode_number}</span>`)}</div></div>` : null}</div></div></div></div></div>`;
}

function App() {
  const [boot, setBoot] = useState(null);
  const [tab, setTab] = useState("home");
  const [detail, setDetail] = useState(null);
  const [toast, setToast] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchType, setSearchType] = useState("all");
  const [searchResults, setSearchResults] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [adForm, setAdForm] = useState({ title: "", description: "", buttonText: "", buttonUrl: "", photoUrl: "" });
  const [uploading, setUploading] = useState(false);
  const [noticeForm, setNoticeForm] = useState({ text: "", link: "" });
  const [proForm, setProForm] = useState({ priceText: "", durationDays: "" });
  const [channelInput, setChannelInput] = useState("");
  const [adminUserQuery, setAdminUserQuery] = useState("");
  const [adminUsers, setAdminUsers] = useState([]);
  const [adminUserLoading, setAdminUserLoading] = useState(false);
  const [adChannelMap, setAdChannelMap] = useState({});

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
          notify("New admin notice");
        }
        return payload;
      });
    } catch (err) {
      setError(err.message || "Load failed");
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
    if (!adminUserQuery.trim()) {
      setAdminUsers(boot?.admin?.recent_users || []);
    }
  }, [boot?.admin?.recent_users, adminUserQuery]);

  useEffect(() => {
    setMenuOpen(false);
  }, [tab]);

  useEffect(() => {
    const firstChannelId = boot?.admin?.ad_channels?.[0]?.id || "";
    if (!firstChannelId) {
      return;
    }
    setAdChannelMap((current) => {
      const next = { ...current };
      for (const ad of boot?.admin?.pending_ads || []) {
        if (!next[ad.id]) {
          next[ad.id] = firstChannelId;
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
      notify(err.message || "Detail failed");
    } finally {
      setBusy(false);
    }
  }

  async function favorite(item) {
    try {
      setBusy(true);
      const payload = await api("/api/favorites/toggle", { method: "POST", body: { contentType: item.content_type, contentRef: item.id } });
      notify(payload.active ? "Saved" : "Removed");
      await loadBoot(true);
      await refreshDetail(item);
    } catch (err) {
      notify(err.message || "Save failed");
    } finally {
      setBusy(false);
    }
  }

  async function react(item, reaction) {
    try {
      setBusy(true);
      await api("/api/reactions", { method: "POST", body: { contentType: item.content_type, contentRef: item.id, reaction } });
      notify("Reaction updated");
      await loadBoot(true);
      await refreshDetail(item);
    } catch (err) {
      notify(err.message || "Reaction failed");
    } finally {
      setBusy(false);
    }
  }

  async function search() {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }
    try {
      setSearchLoading(true);
      const payload = await api(`/api/search?q=${encodeURIComponent(searchQuery.trim())}&type=${encodeURIComponent(searchType)}`);
      setSearchResults(payload.items || []);
      notify(`${payload.items?.length || 0} results`);
    } catch (err) {
      notify(err.message || "Search failed");
    } finally {
      setSearchLoading(false);
    }
  }

  async function toggleNotification(key) {
    try {
      await api("/api/notifications/toggle", { method: "POST", body: { key } });
      await loadBoot(true);
      notify("Setting updated");
    } catch (err) {
      notify(err.message || "Update failed");
    }
  }

  async function uploadImage(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      setUploading(true);
      const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(new Error("File read failed"));
        reader.readAsDataURL(file);
      });
      const payload = await api("/api/upload-image", { method: "POST", body: { dataUrl } });
      setAdForm((current) => ({ ...current, photoUrl: payload.url }));
      notify("Image uploaded");
    } catch (err) {
      notify(err.message || "Upload failed");
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
      notify("Ad sent for review");
    } catch (err) {
      notify(err.message || "Ad failed");
    } finally {
      setBusy(false);
    }
  }

  async function setContentMode(mode) {
    try {
      await api("/api/admin/content-mode", { method: "POST", body: { mode } });
      await loadBoot(true);
      notify("Media mode updated");
    } catch (err) {
      notify(err.message || "Mode update failed");
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
      notify(clear ? "Notice cleared" : "Notice saved");
    } catch (err) {
      notify(err.message || "Notice update failed");
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
      notify(action === "approve" ? "Payment approved" : "Payment rejected");
    } catch (err) {
      notify(err.message || "Payment review failed");
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
          channelId: action === "approve" ? (adChannelMap[adId] || "") : "",
        },
      });
      await loadBoot(true);
      notify(action === "approve" ? "Ad approved" : "Ad rejected");
    } catch (err) {
      notify(err.message || "Ad review failed");
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
      notify("PRO settings saved");
    } catch (err) {
      notify(err.message || "PRO settings failed");
    } finally {
      setBusy(false);
    }
  }

  async function createAdChannel() {
    try {
      setBusy(true);
      const payload = await api("/api/admin/ad-channels/create", {
        method: "POST",
        body: { channelRef: channelInput },
      });
      setChannelInput("");
      await loadBoot(true);
      notify(payload.created ? "Channel added" : "Channel already exists");
    } catch (err) {
      notify(err.message || "Channel add failed");
    } finally {
      setBusy(false);
    }
  }

  async function deleteAdChannel(channelId) {
    try {
      setBusy(true);
      await api("/api/admin/ad-channels/delete", {
        method: "POST",
        body: { channelId },
      });
      await loadBoot(true);
      notify("Channel removed");
    } catch (err) {
      notify(err.message || "Channel delete failed");
    } finally {
      setBusy(false);
    }
  }

  async function searchAdminUsers() {
    try {
      setAdminUserLoading(true);
      const payload = await api(`/api/admin/users/search?q=${encodeURIComponent(adminUserQuery.trim())}`);
      setAdminUsers(payload.items || []);
      notify(`${payload.items?.length || 0} users`);
    } catch (err) {
      notify(err.message || "User search failed");
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
      setAdminUsers((current) => current.map((item) => item.id === userId ? payload.item : item));
      await loadBoot(true);
      notify(enabled ? "PRO enabled" : "PRO disabled");
    } catch (err) {
      notify(err.message || "PRO update failed");
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
      setAdminUsers((current) => current.map((item) => item.id === userId ? payload.item : item));
      await loadBoot(true);
      notify(enabled ? "Admin enabled" : "Admin removed");
    } catch (err) {
      notify(err.message || "Admin update failed");
    } finally {
      setBusy(false);
    }
  }

  const navItems = useMemo(() => boot?.user?.is_admin ? [...NAV, { key: "admin", label: "Admin", icon: "shield" }] : NAV, [boot]);
  const primaryNavItems = useMemo(() => navItems.filter((item) => PRIMARY_NAV_KEYS.includes(item.key)), [navItems]);
  const secondaryNavItems = useMemo(() => navItems.filter((item) => !PRIMARY_NAV_KEYS.includes(item.key)), [navItems]);
  const secondaryActive = secondaryNavItems.some((item) => item.key === tab);

  if (loading) {
    return html`<div className="loader-wrap"><div className="loader-card"><div className="spinner"></div><div className="eyebrow">Mini App</div><h2 className="headline" style=${{ fontSize: "30px", margin: "8px 0 10px" }}>Loading catalog</h2><p className="subheadline">Data is syncing from Telegram bot storage.</p></div></div>`;
  }

  if (!boot || error) {
    return html`<div className="loader-wrap"><div className="loader-card"><div className="eyebrow">Mini App</div><h2 className="headline" style=${{ fontSize: "30px", margin: "8px 0 10px" }}>Connection failed</h2><p className="subheadline">${error || "Unknown error"}</p><div className="hero-actions"><button className="button" onClick=${() => loadBoot(false)}>Retry</button></div></div></div>`;
  }

  const sections = boot.sections || {};
  const admin = boot.admin || {};

  return html`
    <div className="app-shell">
      ${toast ? html`<div className="toast">${toast}</div>` : null}
      ${boot.notice?.text ? html`<section className="notice-bar"><div className="notice-copy"><div className="eyebrow">Admin notice</div><strong>${boot.notice.text}</strong><span className="muted">${dateText(boot.notice.updated_at)}</span></div><div className="notice-actions">${boot.notice.link ? html`<button className="button secondary" onClick=${() => (tg?.openLink ? tg.openLink(boot.notice.link) : window.open(boot.notice.link, "_blank"))}>Open</button>` : null}</div></section>` : null}
      <header className="topbar">
        <div className="brand">
          <div className="eyebrow">Telegram Mini App</div>
          <h1 className="headline">Kino katalog</h1>
          <p className="subheadline">Clean interface for search, favorites, PRO, ads, and admin overview.</p>
        </div>
        <div className="status-cluster">
          <div className="pill">${icon("user")}${boot.user.full_name || boot.user.username || boot.user.id}</div>
          <div className="pill">${icon("crown")}${boot.user.is_pro ? "PRO active" : "Standard"}</div>
          <div className="pill muted">${icon("shield")}${boot.settings.content_mode_label}</div>
        </div>
      </header>

      ${tab === "home" ? html`
        <section className="hero">
          <div className="panel hero-main">
            <div className="eyebrow">Overview</div>
            <h2 style=${{ margin: "8px 0 10px", fontSize: "32px" }}>One app for the whole bot</h2>
            <p className="subheadline">Browse content, open it in the bot, track PRO, and manage ads from one screen.</p>
            <div className="hero-actions">
              <button className="button" onClick=${() => setTab("search")}>Search catalog</button>
              <button className="button secondary" onClick=${() => sendToBot({ action: "open_pro" }, notify)}>Continue in bot</button>
            </div>
          </div>
          <div className="panel hero-side">
            <div className="metric-grid">
              <div className="metric-card"><div className="metric-label">Saved</div><div className="metric-value">${sections.favorites?.length || 0}</div></div>
              <div className="metric-card"><div className="metric-label">Top cards</div><div className="metric-value">${sections.top_viewed?.length || 0}</div></div>
              <div className="metric-card"><div className="metric-label">Media mode</div><div className="metric-value">${boot.settings.content_mode_label}</div></div>
              <div className="metric-card"><div className="metric-label">Pending ads</div><div className="metric-value">${admin.pending_ads_count || 0}</div></div>
            </div>
          </div>
        </section>

        <section className="section">
          <${SectionTitle} iconName="grid" title="Recent movies" copy="Newest movie cards from the database" />
          ${sections.recent_movies?.length ? html`<div className="content-grid">${sections.recent_movies.map((item) => html`<${Card} key=${item.id} item=${item} onOpen=${openDetail} onFavorite=${favorite} onReact=${react} />`)}</div>` : html`<${Empty} title="No movies yet" copy="Admins have not added movie cards yet." />`}
        </section>

        <section className="section">
          <${SectionTitle} iconName="play" title="Recent serials" copy="Latest serial previews and selectors" />
          ${sections.recent_serials?.length ? html`<div className="content-grid">${sections.recent_serials.map((item) => html`<${Card} key=${item.id} item=${item} onOpen=${openDetail} onFavorite=${favorite} onReact=${react} />`)}</div>` : html`<${Empty} title="No serials yet" copy="Serial cards will appear here." />`}
        </section>

        <section className="section">
          <${SectionTitle} iconName="stats" title="Top viewed" copy="Most opened content in the bot" />
          ${sections.top_viewed?.length ? html`<div className="content-grid">${sections.top_viewed.map((item) => html`<${Card} key=${item.id} item=${item} onOpen=${openDetail} onFavorite=${favorite} onReact=${react} />`)}</div>` : html`<${Empty} title="No stats yet" copy="Views will appear after users start opening content." />`}
        </section>
      ` : null}

      ${tab === "search" ? html`
        <section className="section">
          <${SectionTitle} iconName="search" title="Search" copy="Search by code, title, or description" />
          <div className="panel search-box">
            <div className="field"><label>Query</label><input className="input" value=${searchQuery} onInput=${(event) => setSearchQuery(event.target.value)} placeholder="Movie code or title" /></div>
            <div className="chips">${SEARCH_TYPES.map((item) => html`<button className=${joinClass("chip", searchType === item.key && "active")} key=${item.key} onClick=${() => setSearchType(item.key)}>${item.label}</button>`)}</div>
            <div className="hero-actions"><button className="button" onClick=${search} disabled=${searchLoading}>${searchLoading ? "Searching..." : "Search"}</button></div>
          </div>
        </section>
        <section className="section">
          ${searchResults?.length ? html`<div className="content-grid">${searchResults.map((item) => html`<${Card} key=${item.id} item=${item} onOpen=${openDetail} onFavorite=${favorite} onReact=${react} />`)}</div>` : html`<${Empty} title="No results yet" copy="Start with a code or title search." />`}
        </section>
      ` : null}

      ${tab === "saved" ? html`
        <section className="section">
          <${SectionTitle} iconName="bookmark" title="Saved content" copy="Favorites synced with the bot" />
          ${sections.favorites?.length ? html`<div className="content-grid">${sections.favorites.map((item) => html`<${Card} key=${item.id} item=${item} onOpen=${openDetail} onFavorite=${favorite} onReact=${react} />`)}</div>` : html`<${Empty} title="Saved list is empty" copy="Use Save on any card to keep it here." />`}
        </section>
      ` : null}

      ${tab === "pro" ? html`
        <section className="section">
          <${SectionTitle} iconName="crown" title="PRO access" copy="Single tariff managed by admins" />
          <div className="hero">
            <div className="panel hero-main">
              <div className="eyebrow">Current state</div>
              <h2 style=${{ margin: "8px 0 10px", fontSize: "32px" }}>${boot.user.is_pro ? "PRO active" : "Upgrade required"}</h2>
              <p className="subheadline">Price: ${boot.settings.pro_price_text}<br />Duration: ${boot.settings.pro_duration_days} days<br />Until: ${boot.user.pro_until || "—"}</p>
              <div className="hero-actions">
                <button className="button" onClick=${() => sendToBot({ action: "open_pro" }, notify)}>Continue in bot</button>
                <button className="button secondary" onClick=${() => copyText(boot.payment.code, notify)}>Copy code</button>
              </div>
            </div>
            <div className="panel hero-side">
              <div className="list">
                <div className="list-card"><div className="metric-label">Payment code</div><div style=${{ fontWeight: 800, fontSize: "24px" }}>${boot.payment.code}</div></div>
                <div className="list-card"><div className="metric-label">Telegram ID</div><div style=${{ fontWeight: 800, fontSize: "24px" }}>${boot.user.id}</div></div>
              </div>
              <div className="hero-actions">${(boot.payment.links || []).map((link, index) => html`<button className="button secondary" key=${link} onClick=${() => (tg?.openLink ? tg.openLink(link) : window.open(link, "_blank"))}>${icon("link")}Payment ${index + 1}</button>`)}</div>
            </div>
          </div>
        </section>
      ` : null}

      ${tab === "ads" ? html`
        <section className="section">
          <${SectionTitle} iconName="grid" title="My ads" copy="Latest ad requests and statuses" />
          ${boot.ads?.mine?.length ? html`<div className="list">${boot.ads.mine.map((item) => html`<div className="list-card" key=${item.id}><div className="list-row"><div><div style=${{ fontWeight: 700 }}>${item.title}</div><div className="muted">${item.status} · ${dateText(item.created_at)}</div></div><div className="tag">${item.status}</div></div><p className="muted" style=${{ marginBottom: 0 }}>${item.description}</p></div>`)}</div>` : html`<${Empty} title="No ads yet" copy="Your submitted ads will appear here." />`}
        </section>

        <section className="section">
          <${SectionTitle} iconName="megaphone" title="Create ad" copy="Publish if your PRO is active" />
          ${boot.ads?.can_create ? html`<div className="hero"><form className="panel search-box" onSubmit=${submitAd}><div className="form-grid"><div className="field"><label>Title</label><input className="input" value=${adForm.title} onInput=${(event) => setAdForm((current) => ({ ...current, title: event.target.value }))} placeholder="Ad title" /></div><div className="field"><label>Button text</label><input className="input" value=${adForm.buttonText} onInput=${(event) => setAdForm((current) => ({ ...current, buttonText: event.target.value }))} placeholder="Optional" /></div></div><div className="field"><label>Description</label><textarea className="textarea" value=${adForm.description} onInput=${(event) => setAdForm((current) => ({ ...current, description: event.target.value }))} placeholder="Description"></textarea></div><div className="field"><label>Button URL</label><input className="input" value=${adForm.buttonUrl} onInput=${(event) => setAdForm((current) => ({ ...current, buttonUrl: event.target.value }))} placeholder="https://..." /></div><div className="hero-actions"><button className="button" type="submit" disabled=${busy}>Send for review</button><button className="button ghost" type="button" onClick=${() => setAdForm({ title: "", description: "", buttonText: "", buttonUrl: "", photoUrl: "" })}>Reset</button></div></form><div className="panel hero-side"><div className="upload-box"><div className="metric-label">Image</div>${adForm.photoUrl ? html`<img className="upload-preview" src=${adForm.photoUrl} alt="ad preview" />` : html`<div className="upload-preview" style=${{ display: "grid", placeItems: "center" }}>Preview</div>`}<input className="input" type="file" accept="image/*" onChange=${uploadImage} /><div className="muted">${uploading ? "Uploading..." : "Upload an optional image for the ad."}</div></div></div></div>` : html`<div className="panel hero-main"><div className="eyebrow">Access locked</div><h2 style=${{ margin: "8px 0 10px", fontSize: "32px" }}>PRO required</h2><p className="subheadline">Ad publishing is available only for PRO users.</p><div className="hero-actions"><button className="button" onClick=${() => sendToBot({ action: "open_pro" }, notify)}>Activate PRO</button></div></div>`}
        </section>
      ` : null}

      ${tab === "profile" ? html`
        <section className="section">
          <${SectionTitle} iconName="grid" title="Profile" copy="Account and notification settings" />
          <div className="hero">
            <div className="panel hero-main">
              <div className="eyebrow">Account</div>
              <h2 style=${{ margin: "8px 0 10px", fontSize: "32px" }}>${boot.user.full_name || boot.user.username || boot.user.id}</h2>
              <p className="subheadline">Telegram ID: ${boot.user.id}<br />PRO: ${boot.user.is_pro ? "active" : "inactive"}<br />Admin: ${boot.user.is_admin ? "yes" : "no"}</p>
              <div className="hero-actions"><button className="button secondary" onClick=${() => copyText(boot.user.id, notify)}>Copy ID</button><button className="button ghost" onClick=${() => sendToBot({ action: "open_notifications" }, notify)}>Open bot settings</button></div>
            </div>
            <div className="panel hero-side">
              <div className="list-card"><div className="list-row"><span>New content alerts</span><button className=${joinClass("button", boot.notifications.new_content ? "success" : "ghost")} onClick=${() => toggleNotification("new_content")}>${boot.notifications.new_content ? "On" : "Off"}</button></div></div>
              <div className="list-card"><div className="list-row"><span>PRO updates</span><button className=${joinClass("button", boot.notifications.pro_updates ? "success" : "ghost")} onClick=${() => toggleNotification("pro_updates")}>${boot.notifications.pro_updates ? "On" : "Off"}</button></div></div>
              <div className="list-card"><div className="list-row"><span>Ad updates</span><button className=${joinClass("button", boot.notifications.ads_updates ? "success" : "ghost")} onClick=${() => toggleNotification("ads_updates")}>${boot.notifications.ads_updates ? "On" : "Off"}</button></div></div>
            </div>
          </div>
        </section>
      ` : null}

      ${tab === "admin" && boot.user.is_admin ? html`
        <section className="section">
          <${SectionTitle} iconName="shield" title="Admin control" copy="Live counters and global media mode" />
          <div className="hero"><div className="panel hero-side"><div className="metric-grid"><div className="metric-card"><div className="metric-label">Users</div><div className="metric-value">${admin.total_users || 0}</div></div><div className="metric-card"><div className="metric-label">PRO users</div><div className="metric-value">${admin.total_pro_users || 0}</div></div><div className="metric-card"><div className="metric-label">Movies</div><div className="metric-value">${admin.total_movies || 0}</div></div><div className="metric-card"><div className="metric-label">Serials</div><div className="metric-value">${admin.total_serials || 0}</div></div></div></div><div className="panel hero-main"><div className="eyebrow">Media mode</div><h2 style=${{ margin: "8px 0 10px", fontSize: "32px" }}>${boot.settings.content_mode_label}</h2><p className="subheadline">This toggle affects every user and every admin.</p><div className="hero-actions"><button className="button secondary" onClick=${() => setContentMode("private")}>Private</button><button className="button secondary" onClick=${() => setContentMode("public")}>Public</button></div></div></div>
        </section>
        <section className="section">
          <${SectionTitle} iconName="user" title="Users" copy="Search users and manage PRO or admin access." />
          <div className="panel search-box">
            <div className="form-grid">
              <div className="field"><label>User search</label><input className="input" value=${adminUserQuery} onInput=${(event) => setAdminUserQuery(event.target.value)} placeholder="Telegram ID or full name" /></div>
              <div className="hero-actions"><button className="button" onClick=${searchAdminUsers} disabled=${adminUserLoading}>${adminUserLoading ? "Searching..." : "Search users"}</button><button className="button ghost" onClick=${() => { setAdminUserQuery(""); setAdminUsers(admin.recent_users || []); }} disabled=${adminUserLoading}>Recent</button></div>
            </div>
          </div>
          ${(adminUsers || []).length ? html`<div className="list" style=${{ marginTop: "14px" }}>${(adminUsers || []).map((item) => html`<div className="list-card" key=${item.id}><div className="list-row"><div><div style=${{ fontWeight: 700 }}>${item.full_name || `User ${item.id}`}</div><div className="muted">ID ${item.id} · Joined ${dateText(item.joined_at)}</div></div><div className="chips"><span className="tag">${item.is_pro ? "PRO" : "Free"}</span><span className="tag">${item.is_admin ? "Admin" : "User"}</span></div></div><div className="muted" style=${{ marginTop: "10px" }}>PRO until: ${item.pro_until || "—"}</div>${item.is_seed_admin ? html`<div className="muted" style=${{ marginTop: "6px" }}>Seed admin: managed from env</div>` : null}<div className="hero-actions"><button className=${joinClass("button", item.is_pro ? "danger" : "success")} onClick=${() => setUserPro(item.id, !item.is_pro)} disabled=${busy}>${item.is_pro ? "Disable PRO" : "Enable PRO"}</button><button className=${joinClass("button", item.is_admin ? "danger" : "secondary")} onClick=${() => setUserAdmin(item.id, !item.is_admin)} disabled=${busy || (item.id === boot.user.id && item.is_admin) || item.is_seed_admin}>${item.is_admin ? "Remove admin" : "Make admin"}</button></div></div>`)}</div>` : html`<div style=${{ marginTop: "14px" }}><${Empty} title="No users found" copy="Search by Telegram ID or full name." /></div>`}
        </section>
        <section className="section">
          <${SectionTitle} iconName="grid" title="Site notice" copy="Mini App users will see this notice on refresh and live polling." />
          <div className="panel search-box">
            <div className="field"><label>Notice text</label><textarea className="textarea" value=${noticeForm.text} onInput=${(event) => setNoticeForm((current) => ({ ...current, text: event.target.value }))} placeholder="Short announcement for the Mini App"></textarea></div>
            <div className="field"><label>Notice link</label><input className="input" value=${noticeForm.link} onInput=${(event) => setNoticeForm((current) => ({ ...current, link: event.target.value }))} placeholder="Optional https://... or /app/" /></div>
            <div className="hero-actions"><button className="button" onClick=${() => saveNotice(false)} disabled=${busy}>Save notice</button><button className="button ghost" onClick=${() => saveNotice(true)} disabled=${busy}>Clear</button></div>
          </div>
        </section>
        <section className="section">
          <${SectionTitle} iconName="crown" title="PRO settings" copy="Single tariff used by bot and Mini App." />
          <div className="panel search-box">
            <div className="form-grid">
              <div className="field"><label>Price text</label><input className="input" value=${proForm.priceText} onInput=${(event) => setProForm((current) => ({ ...current, priceText: event.target.value }))} placeholder="12000 so'm" /></div>
              <div className="field"><label>Duration days</label><input className="input" type="number" min="1" max="3650" value=${proForm.durationDays} onInput=${(event) => setProForm((current) => ({ ...current, durationDays: event.target.value }))} placeholder="30" /></div>
            </div>
            <div className="hero-actions"><button className="button" onClick=${saveProSettings} disabled=${busy}>Save PRO</button></div>
          </div>
        </section>
        <section className="section">
          <${SectionTitle} iconName="megaphone" title="Ad channels" copy="Channels used when pending ads are approved." />
          <div className="hero">
            <div className="panel search-box">
              <div className="field"><label>Channel</label><input className="input" value=${channelInput} onInput=${(event) => setChannelInput(event.target.value)} placeholder="@channel or -100..." /></div>
              <div className="hero-actions"><button className="button" onClick=${createAdChannel} disabled=${busy || !channelInput.trim()}>Add channel</button></div>
            </div>
            <div className="panel hero-side">
              ${(admin.ad_channels || []).length ? html`<div className="list">${(admin.ad_channels || []).map((channel) => html`<div className="list-card" key=${channel.id}><div className="list-row"><div><div style=${{ fontWeight: 700 }}>${channel.title}</div><div className="muted">${channel.channel_ref}</div></div><button className="button danger" onClick=${() => deleteAdChannel(channel.id)} disabled=${busy}>Remove</button></div></div>`)}</div>` : html`<${Empty} title="No ad channels" copy="Add a target channel for ad approvals." />`}
            </div>
          </div>
        </section>
        <section className="section"><${SectionTitle} iconName="stats" title="Pending payments" copy="Latest PRO requests waiting for review" />${admin.pending_payments?.length ? html`<div className="list">${admin.pending_payments.map((item) => html`<div className="list-card" key=${item.id}><div className="list-row"><div><div style=${{ fontWeight: 700 }}>User ${item.user_tg_id}</div><div className="muted">${item.payment_code || "No code"} · ${dateText(item.created_at)}</div></div><div className="tag">${item.status}</div></div>${item.comment ? html`<p className="muted">${item.comment}</p>` : null}<div className="hero-actions"><button className="button success" onClick=${() => reviewPayment(item.id, "approve")} disabled=${busy}>Approve</button><button className="button danger" onClick=${() => reviewPayment(item.id, "reject")} disabled=${busy}>Reject</button></div></div>`)}</div>` : html`<${Empty} title="No pending payments" copy="All PRO requests are processed." />`}</section>
        <section className="section"><${SectionTitle} iconName="megaphone" title="Pending ads" copy="Ads waiting for moderator action" />${admin.pending_ads?.length ? html`<div className="list">${admin.pending_ads.map((item) => html`<div className="list-card" key=${item.id}><div className="list-row"><div><div style=${{ fontWeight: 700 }}>${item.title}</div><div className="muted">User ${item.user_tg_id} · ${dateText(item.created_at)}</div></div><div className="tag">${item.status}</div></div><p className="muted">${item.description}</p><div className="field"><label>Channel</label><select className="input" value=${adChannelMap[item.id] || ""} onChange=${(event) => setAdChannelMap((current) => ({ ...current, [item.id]: event.target.value }))}>${(admin.ad_channels || []).map((channel) => html`<option value=${channel.id} key=${channel.id}>${channel.title}</option>`)}</select></div><div className="hero-actions"><button className="button success" onClick=${() => reviewAd(item.id, "approve")} disabled=${busy || !(admin.ad_channels || []).length}>Approve</button><button className="button danger" onClick=${() => reviewAd(item.id, "reject")} disabled=${busy}>Reject</button></div></div>`)}</div>` : html`<${Empty} title="No pending ads" copy="Ad moderation queue is empty." />`}</section>
      ` : null}

      <nav className="bottom-nav">
        <div className="bottom-nav-inner">
          ${primaryNavItems.map((item) => html`<button className=${joinClass("nav-item", tab === item.key && "active")} key=${item.key} onClick=${() => setTab(item.key)}>${icon(item.icon)}<span className="nav-label">${item.label}</span></button>`)}
          <button className=${joinClass("nav-item", (menuOpen || secondaryActive) && "active")} onClick=${() => setMenuOpen((current) => !current)}>${icon("menu")}<span className="nav-label">More</span></button>
        </div>
      </nav>
      ${menuOpen ? html`<div className="menu-backdrop" onClick=${() => setMenuOpen(false)}><div className="menu-sheet" onClick=${(event) => event.stopPropagation()}><div className="menu-sheet-header"><div><div className="eyebrow">Quick access</div><h3 style=${{ margin: "8px 0 0", fontSize: "24px" }}>More</h3></div><button className="icon-button" onClick=${() => setMenuOpen(false)}>${icon("close")}</button></div><div className="menu-grid">${secondaryNavItems.map((item) => html`<button className=${joinClass("menu-card", tab === item.key && "active")} key=${item.key} onClick=${() => setTab(item.key)}><div className="menu-card-icon">${icon(item.icon)}</div><div className="menu-card-label">${item.label}</div></button>`)}</div></div></div>` : null}
      <${DetailSheet} item=${detail} onClose=${() => setDetail(null)} onFavorite=${favorite} onReact=${react} onBotOpen=${(item) => sendToBot(item.open_payload, notify)} />
    </div>
  `;
}

createRoot(document.getElementById("root")).render(html`<${App} />`);
