const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export function getApiBase() {
  return API_BASE.replace(/\/+$/, "");
}

export function getTelegramInitData() {
  const fromTelegram = window.Telegram?.WebApp?.initData || "";
  if (fromTelegram) {
    localStorage.setItem("tg_init_data", fromTelegram);
    return fromTelegram;
  }
  return localStorage.getItem("tg_init_data") || "";
}

async function request(path, { method = "GET", body, initData, signal } = {}) {
  const headers = {
    "Content-Type": "application/json"
  };
  if (initData) {
    headers["X-Telegram-Init-Data"] = initData;
  }

  const res = await fetch(`${getApiBase()}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
    signal
  });

  const text = await res.text();
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = {};
    }
  }

  if (!res.ok) {
    const error = new Error(data?.detail?.message || data?.detail || "Request failed");
    error.status = res.status;
    error.payload = data;
    throw error;
  }
  return data;
}

export function fetchBootstrap(initData) {
  return request("/api/bootstrap", { initData });
}

export function fetchContent({ initData, query, contentType, signal }) {
  const q = encodeURIComponent(query || "");
  const type = encodeURIComponent(contentType || "all");
  return request(`/api/content?q=${q}&content_type=${type}&limit=120`, { initData, signal });
}

export function fetchFavorites(initData) {
  return request("/api/favorites", { initData });
}

export function toggleFavorite({ initData, contentType, contentRef }) {
  return request("/api/favorites/toggle", {
    method: "POST",
    initData,
    body: {
      content_type: contentType,
      content_ref: contentRef
    }
  });
}

export function fetchProfile(initData) {
  return request("/api/profile", { initData });
}

export function fetchHistory(initData) {
  return request("/api/history?limit=30", { initData });
}

export function fetchContentDetail({ initData, contentType, contentRef }) {
  return request(`/api/content/${encodeURIComponent(contentType)}/${encodeURIComponent(contentRef)}`, { initData });
}

export function fetchRecommendations({ initData, contentType, contentRef, limit = 12 }) {
  return request(
    `/api/recommendations?content_type=${encodeURIComponent(contentType)}&content_ref=${encodeURIComponent(contentRef)}&limit=${encodeURIComponent(limit)}`,
    { initData }
  );
}

export function setReaction({ initData, contentType, contentRef, reaction }) {
  return request("/api/reactions/set", {
    method: "POST",
    initData,
    body: {
      content_type: contentType,
      content_ref: contentRef,
      reaction
    }
  });
}

export function fetchComments({ initData, contentType, contentRef }) {
  return request(
    `/api/comments?content_type=${encodeURIComponent(contentType)}&content_ref=${encodeURIComponent(contentRef)}&limit=80`,
    { initData }
  );
}

export function addComment({ initData, contentType, contentRef, text }) {
  return request("/api/comments/add", {
    method: "POST",
    initData,
    body: {
      content_type: contentType,
      content_ref: contentRef,
      text
    }
  });
}

export function trackDownload({ initData, contentType, contentRef }) {
  return request("/api/downloads/track", {
    method: "POST",
    initData,
    body: {
      content_type: contentType,
      content_ref: contentRef
    }
  });
}

export function buildMediaUrl(fileId, initData) {
  const encodedId = encodeURIComponent(fileId);
  const encodedInit = encodeURIComponent(initData || "");
  return `${getApiBase()}/api/media/file?file_id=${encodedId}&init_data=${encodedInit}`;
}
