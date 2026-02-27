const API_BASE = import.meta.env.VITE_API_BASE || ''

const responseCache = new Map()

function now() {
	return Date.now()
}

function getCached(cacheKey) {
	const row = responseCache.get(cacheKey)
	if (!row) return null
	if (row.expiresAt <= now()) {
		responseCache.delete(cacheKey)
		return null
	}
	return row.value
}

function setCached(cacheKey, value, ttlMs) {
	if (!cacheKey || ttlMs <= 0) return
	responseCache.set(cacheKey, {
		value,
		expiresAt: now() + ttlMs,
	})
}

function clearCacheByPrefix(prefix) {
	if (!prefix) {
		responseCache.clear()
		return
	}
	for (const key of responseCache.keys()) {
		if (key.startsWith(prefix)) {
			responseCache.delete(key)
		}
	}
}

export function getApiBase() {
	if (!API_BASE) {
		return window.location.origin.replace(/\/+$/, '')
	}
	return API_BASE.replace(/\/+$/, '')
}

export function getTelegramInitData() {
	const fromTelegram = window.Telegram?.WebApp?.initData || ''
	if (fromTelegram) {
		localStorage.setItem('tg_init_data', fromTelegram)
		return fromTelegram
	}
	return localStorage.getItem('tg_init_data') || ''
}

async function request(path, { method = 'GET', body, initData, signal, cacheTtlMs = 0 } = {}) {
	const normalizedMethod = String(method || 'GET').toUpperCase()
	const cacheKey = normalizedMethod === 'GET' && cacheTtlMs > 0
		? `${normalizedMethod}:${path}:${String(initData || '').slice(0, 120)}`
		: ''

	if (cacheKey) {
		const cached = getCached(cacheKey)
		if (cached) return cached
	}

	const headers = {
		'Content-Type': 'application/json',
	}
	if (initData) {
		headers['X-Telegram-Init-Data'] = initData
	}

	const res = await fetch(`${getApiBase()}${path}`, {
		method: normalizedMethod,
		headers,
		body: body ? JSON.stringify(body) : undefined,
		signal,
	})

	const text = await res.text()
	let data = {}
	if (text) {
		try {
			data = JSON.parse(text)
		} catch {
			data = {}
		}
	}

	if (!res.ok) {
		const error = new Error(data?.detail?.message || data?.detail || 'Request failed')
		error.status = res.status
		error.payload = data
		throw error
	}

	if (cacheKey) {
		setCached(cacheKey, data, cacheTtlMs)
	}
	return data
}

export function invalidateApiCache(prefix = '') {
	clearCacheByPrefix(prefix)
}

export function fetchBootstrap(initData) {
	return request('/api/bootstrap', { initData, cacheTtlMs: 20000 })
}

export function fetchContent({ initData, query, contentType, signal }) {
	const q = encodeURIComponent(query || '')
	const type = encodeURIComponent(contentType || 'all')
	return request(`/api/content?q=${q}&content_type=${type}&limit=120`, {
		initData,
		signal,
		cacheTtlMs: 8000,
	})
}

export function fetchFavorites(initData) {
	return request('/api/favorites', { initData, cacheTtlMs: 6000 })
}

export async function toggleFavorite({ initData, contentType, contentRef }) {
	const result = await request('/api/favorites/toggle', {
		method: 'POST',
		initData,
		body: {
			content_type: contentType,
			content_ref: contentRef,
		},
	})
	clearCacheByPrefix('GET:/api/favorites')
	return result
}

export function fetchProfile(initData) {
	return request('/api/profile', { initData, cacheTtlMs: 12000 })
}

export function fetchHistory(initData) {
	return request('/api/history?limit=30', { initData, cacheTtlMs: 12000 })
}

export function fetchContentDetail({ initData, contentType, contentRef }) {
	return request(`/api/content/${encodeURIComponent(contentType)}/${encodeURIComponent(contentRef)}`, {
		initData,
		cacheTtlMs: 12000,
	})
}

export function fetchRecommendations({ initData, contentType, contentRef, limit = 12 }) {
	return request(
		`/api/recommendations?content_type=${encodeURIComponent(contentType)}&content_ref=${encodeURIComponent(contentRef)}&limit=${encodeURIComponent(limit)}`,
		{ initData, cacheTtlMs: 10000 },
	)
}

export function fetchRecommendationsFeed({ initData, limit = 12 }) {
	return request(`/api/recommendations/feed?limit=${encodeURIComponent(limit)}`, {
		initData,
		cacheTtlMs: 12000,
	})
}

export function setReaction({ initData, contentType, contentRef, reaction }) {
	return request('/api/reactions/set', {
		method: 'POST',
		initData,
		body: {
			content_type: contentType,
			content_ref: contentRef,
			reaction,
		},
	})
}

export function fetchComments({ initData, contentType, contentRef }) {
	return request(
		`/api/comments?content_type=${encodeURIComponent(contentType)}&content_ref=${encodeURIComponent(contentRef)}&limit=80`,
		{ initData, cacheTtlMs: 3000 },
	)
}

export function fetchCommentsSorted({ initData, contentType, contentRef, sort = 'new' }) {
	return request(
		`/api/comments?content_type=${encodeURIComponent(contentType)}&content_ref=${encodeURIComponent(contentRef)}&limit=120&sort=${encodeURIComponent(sort)}`,
		{ initData, cacheTtlMs: 3000 },
	)
}

export function addComment({ initData, contentType, contentRef, text, parentCommentId = null }) {
	return request('/api/comments/add', {
		method: 'POST',
		initData,
		body: {
			content_type: contentType,
			content_ref: contentRef,
			text,
			parent_comment_id: parentCommentId,
		},
	})
}

export function deleteComment({ initData, commentId }) {
	return request(`/api/comments/${encodeURIComponent(commentId)}`, {
		method: 'DELETE',
		initData,
	})
}

export function setCommentReaction({ initData, commentId, reaction }) {
	return request('/api/comments/reactions/set', {
		method: 'POST',
		initData,
		body: {
			comment_id: commentId,
			reaction,
		},
	})
}

export function trackDownload({ initData, contentType, contentRef }) {
	return request('/api/downloads/track', {
		method: 'POST',
		initData,
		body: {
			content_type: contentType,
			content_ref: contentRef,
		},
	})
}

export function fetchWatchInfo({ initData, contentType, contentRef, episode }) {
	const query = new URLSearchParams({
		content_type: contentType,
		content_ref: contentRef,
	})
	if (episode !== undefined && episode !== null && episode !== '') {
		query.set('episode', String(episode))
	}
	return request(`/api/watch?${query.toString()}`, { initData, cacheTtlMs: 6000 })
}

export function trackWatchProgress({ initData, contentType, contentRef, episodeNumber, positionSeconds, durationSeconds }) {
	return request('/api/watch/progress', {
		method: 'POST',
		initData,
		body: {
			content_type: contentType,
			content_ref: contentRef,
			episode_number: episodeNumber ?? null,
			position_seconds: positionSeconds,
			duration_seconds: durationSeconds,
		},
	})
}

export function fetchContinueWatching(initData) {
	return request('/api/continue?limit=30', { initData, cacheTtlMs: 10000 })
}

export function fetchNotifications({ initData, limit = 20 }) {
	return request(`/api/notifications?limit=${encodeURIComponent(limit)}`, {
		initData,
		cacheTtlMs: 5000,
	})
}

export function markNotificationsRead(initData) {
	return request('/api/notifications/read', {
		method: 'POST',
		initData,
	})
}

export function fetchAdminOverview(initData) {
	return request('/api/admin/overview', { initData, cacheTtlMs: 5000 })
}

export function fetchAdminContent({ initData, contentType = 'all', query = '' }) {
	return request(
		`/api/admin/content?content_type=${encodeURIComponent(contentType)}&q=${encodeURIComponent(query)}&limit=250`,
		{ initData, cacheTtlMs: 5000 },
	)
}

export function adminToggleContent({ initData, contentType, contentRef, isActive }) {
	return request('/api/admin/content/toggle', {
		method: 'POST',
		initData,
		body: {
			content_type: contentType,
			content_ref: contentRef,
			is_active: isActive,
		},
	})
}

export function adminCreateContent({ initData, payload }) {
	return request('/api/admin/content/create', {
		method: 'POST',
		initData,
		body: payload,
	})
}

export function buildMediaUrl(fileId, initData) {
	const encodedId = encodeURIComponent(fileId)
	const encodedInit = encodeURIComponent(initData || '')
	return `${getApiBase()}/api/media/file?file_id=${encodedId}&init_data=${encodedInit}`
}

export function buildStreamUrl(fileId, initData) {
	const encodedId = encodeURIComponent(fileId)
	const encodedInit = encodeURIComponent(initData || '')
	return `${getApiBase()}/api/media/stream?file_id=${encodedId}&init_data=${encodedInit}`
}
