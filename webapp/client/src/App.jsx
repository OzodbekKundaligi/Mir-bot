import { useEffect, useMemo, useRef, useState } from 'react'
import {
	Bell,
	Bookmark,
	Clock3,
	Download,
	Flame,
	Forward,
	Fullscreen,
	Home,
	MessageCircle,
	Moon,
	Pause,
	Play,
	Reply,
	RotateCw,
	Search,
	Send,
	Settings,
	Share2,
	Shield,
	SkipBack,
	Sun,
	ThumbsDown,
	ThumbsUp,
	Trash2,
	UserCircle2,
	Video,
	Volume2,
	X,
} from 'lucide-react'
import {
	addComment,
	adminCreateContent,
	adminToggleContent,
	buildMediaUrl,
	buildStreamUrl,
	deleteComment,
	fetchAdminContent,
	fetchAdminOverview,
	fetchBootstrap,
	fetchCommentsSorted,
	fetchContent,
	fetchFavorites,
	fetchNotifications,
	fetchProfile,
	fetchRecommendations,
	fetchWatchInfo,
	getTelegramInitData,
	markNotificationsRead,
	setCommentReaction,
	setReaction,
	toggleFavorite,
	trackDownload,
	trackWatchProgress,
} from './api'

const T = {
	HOME: 'home',
	SAVED: 'saved',
	HISTORY: 'history',
	PROFILE: 'profile',
	ADMIN: 'admin',
}

const DESCRIPTION_LIMIT_CARD = 140
const DESCRIPTION_LIMIT_WATCH = 280

function parseInitialViewState() {
	const params = new URLSearchParams(window.location.search)
	const rawTab = String(params.get('tab') || '')
		.trim()
		.toLowerCase()
	const tab = Object.values(T).includes(rawTab) ? rawTab : T.HOME
	const contentTypeRaw = String(params.get('type') || '')
		.trim()
		.toLowerCase()
	const contentType =
		contentTypeRaw === 'movie' ||
		contentTypeRaw === 'serial' ||
		contentTypeRaw === 'short'
			? contentTypeRaw
			: 'all'
	const sortRaw = String(params.get('sort') || '')
		.trim()
		.toLowerCase()
	const sortMode =
		sortRaw === 'popular' || sortRaw === 'liked' || sortRaw === 'new'
			? sortRaw
			: 'new'
	return {
		tab,
		query: String(params.get('q') || ''),
		contentType,
		genreFilter: String(params.get('genre') || 'all'),
		sortMode,
	}
}

function formatRelativeTime(isoText) {
	const value = String(isoText || '')
	if (!value) return ''
	const ts = Date.parse(value)
	if (Number.isNaN(ts)) return value
	const diffMs = Date.now() - ts
	const minutes = Math.floor(diffMs / 60000)
	if (minutes < 1) return 'Hozir'
	if (minutes < 60) return `${minutes} daqiqa oldin`
	const hours = Math.floor(minutes / 60)
	if (hours < 24) return `${hours} soat oldin`
	const days = Math.floor(hours / 24)
	if (days < 30) return `${days} kun oldin`
	return value.slice(0, 10)
}

function formatDuration(secondsValue) {
	const total = Math.max(0, Math.floor(Number(secondsValue || 0)))
	const h = Math.floor(total / 3600)
	const m = Math.floor((total % 3600) / 60)
	const s = total % 60
	if (h > 0)
		return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
	return `${m}:${String(s).padStart(2, '0')}`
}

function formatContentKind(contentType) {
	const value = String(contentType || '').trim().toLowerCase()
	if (value === 'serial') return 'Serial'
	if (value === 'short') return 'Short'
	return 'Kino'
}

function normalizeTelegramLink(url) {
	const text = String(url || '').trim()
	if (!text) return ''
	if (text.startsWith('tg://') || text.startsWith('http://') || text.startsWith('https://')) {
		return text
	}
	if (text.startsWith('t.me/')) return `https://${text}`
	return text
}

function clamp(value, min, max) {
	return Math.min(max, Math.max(min, value))
}

function uniqueNonEmpty(values) {
	const out = []
	const seen = new Set()
	for (const value of values) {
		const item = String(value || '').trim()
		if (!item || seen.has(item)) continue
		seen.add(item)
		out.push(item)
	}
	return out
}

function isHttpUrl(value) {
	const text = String(value || '').trim().toLowerCase()
	return text.startsWith('http://') || text.startsWith('https://')
}

function isImageMediaType(value) {
	const media = String(value || '').trim().toLowerCase()
	if (!media) return false
	if (media.startsWith('image/')) return true
	return media === 'photo' || media === 'image' || media === 'sticker'
}

function buildPosterSource(fileId, initData, mediaToken, mediaType = 'photo') {
	const fileRef = String(fileId || '').trim()
	if (!fileRef) return ''
	if (isHttpUrl(fileRef)) return fileRef
	return buildMediaUrl(fileRef, initData, mediaToken, mediaType)
}

function buildPlayableSources(fileId, initData, mediaToken, mediaType = '') {
	const fileRef = String(fileId || '').trim()
	if (!fileRef) return []
	if (isHttpUrl(fileRef)) return [fileRef]
	const init = String(initData || '').trim()
	const token = String(mediaToken || '').trim()
	if (!init && !token) return []
	const sources = []
	if (token) {
		sources.push(buildStreamUrl(fileRef, init, token, mediaType))
		sources.push(buildMediaUrl(fileRef, init, token, mediaType))
	}
	sources.push(buildStreamUrl(fileRef, init, '', mediaType))
	sources.push(buildMediaUrl(fileRef, init, '', mediaType))
	return uniqueNonEmpty(sources)
}

function resolveCardPoster(item, initData, mediaToken) {
	const posterRef = String(item.poster_file_id || item.preview_file_id || '').trim()
	const posterType = String(item.preview_media_type || 'photo').trim()
	if (!posterRef || !isImageMediaType(posterType)) return ''
	return buildPosterSource(posterRef, initData, mediaToken, posterType)
}

function resolvePreviewSources(item, initData, mediaToken) {
	const previewRef = String(item.preview_stream_file_id || item.file_id || '').trim()
	const previewType = String(
		item.preview_stream_file_id ? item.preview_media_type : item.media_type,
	).trim()
	if (!previewRef) return []
	if (!item.preview_stream_file_id && isImageMediaType(previewType)) return []
	return buildPlayableSources(previewRef, initData, mediaToken, previewType)
}

function sortContentRows(items, sortMode) {
	const rows = [...items]
	if (sortMode === 'popular') {
		return rows.sort((a, b) => {
			const aScore =
				Number(a.likes || 0) * 4 +
				Number(a.comments || 0) * 3 +
				Number(a.downloads || 0) +
				Number(a.downloads_tracked || 0) -
				Number(a.dislikes || 0) * 2
			const bScore =
				Number(b.likes || 0) * 4 +
				Number(b.comments || 0) * 3 +
				Number(b.downloads || 0) +
				Number(b.downloads_tracked || 0) -
				Number(b.dislikes || 0) * 2
			return bScore - aScore
		})
	}
	if (sortMode === 'liked') {
		return rows.sort(
			(a, b) =>
				Number(b.likes || 0) -
				Number(b.dislikes || 0) -
				(Number(a.likes || 0) - Number(a.dislikes || 0)),
		)
	}
	return rows.sort((a, b) =>
		String(b.created_at || '').localeCompare(String(a.created_at || '')),
	)
}

function ExpandableText({ text, limit = 140, className = '' }) {
	const [expanded, setExpanded] = useState(false)
	const normalized = String(text || '').trim()

	useEffect(() => {
		setExpanded(false)
	}, [normalized])

	if (!normalized) {
		return <p className={`muted ${className}`.trim()}>Tavsif yo'q</p>
	}

	const isLong = normalized.length > limit
	const shownText =
		expanded || !isLong
			? normalized
			: `${normalized.slice(0, limit).trimEnd()}...`

	return (
		<div className={`expandable ${className}`.trim()}>
			<p>{shownText}</p>
			{isLong ? (
				<button
					type='button'
					className='link-btn'
					onClick={() => setExpanded(v => !v)}
				>
					{expanded ? 'Kamroq' : "Ko'proq"}
				</button>
			) : null}
		</div>
	)
}

function ProfileStat({ label, value }) {
	return (
		<div className='profile-stat'>
			<small>{label}</small>
			<strong>{value}</strong>
		</div>
	)
}

function SubscriptionGate({ missingChannels, onRefresh }) {
	return (
		<section className='panel gate'>
			<h2>Majburiy obuna kerak</h2>
			<div className='channel-list'>
				{missingChannels.map(channel => (
					<div className='channel-item' key={channel.channel_ref}>
						<div>
							<strong>{channel.title}</strong>
							<small>{channel.channel_ref}</small>
						</div>
						{channel.join_url ? (
							<a href={channel.join_url}>Qo'shilish</a>
						) : (
							<span>Link yo'q</span>
						)}
					</div>
				))}
			</div>
			<button className='primary' onClick={onRefresh}>
				Tekshirish
			</button>
		</section>
	)
}

function NotificationPanel({ loading, items, onClose, onOpenItem }) {
	return (
		<section className='panel notify-panel'>
			<header className='notify-head'>
				<h3>Bildirishnomalar</h3>
				<button onClick={onClose}>
					<X size={14} />
				</button>
			</header>
			{loading ? <div className='muted'>Yuklanmoqda...</div> : null}
			<div className='notify-list'>
				{items.length ? (
					items.map(row => (
						<button
							key={row.id}
							className='notify-item'
							onClick={() => onOpenItem(row.item)}
						>
							<div>
								<strong>{row.title}</strong>
								<p>{row.text || row.item?.title || ''}</p>
							</div>
							<small>{formatRelativeTime(row.created_at)}</small>
						</button>
					))
				) : (
					<div className='empty-block'>Yangi bildirishnoma yo'q.</div>
				)}
			</div>
		</section>
	)
}

function SkeletonFeed() {
	return (
		<section className='panel'>
			<div className='skeleton-grid'>
				{Array.from({ length: 8 }).map((_, index) => (
					<div className='skeleton-card' key={index}>
						<div className='skeleton-poster' />
						<div className='skeleton-line w80' />
						<div className='skeleton-line w60' />
						<div className='skeleton-line w40' />
					</div>
				))}
			</div>
		</section>
	)
}

function Card({
	item,
	initData,
	mediaToken,
	fav,
	onWatch,
	onFav,
	onDownload,
	onShare,
}) {
	const img = resolveCardPoster(item, initData, mediaToken)
	const [hovered, setHovered] = useState(false)
	const [imageFailed, setImageFailed] = useState(false)
	const [previewFailed, setPreviewFailed] = useState(false)
	const previewSources = useMemo(
		() => resolvePreviewSources(item, initData, mediaToken),
		[
			item.preview_stream_file_id,
			item.file_id,
			item.preview_media_type,
			item.media_type,
			initData,
			mediaToken,
		],
	)
	const [previewIndex, setPreviewIndex] = useState(0)
	const previewSrc = previewSources[previewIndex] || ''
	const showImage = Boolean(img && !imageFailed)
	const showStaticPreview = Boolean(!showImage && previewSrc && !previewFailed)
	const genreLabel = (item.genres || []).slice(0, 2).join(', ')

	useEffect(() => {
		setImageFailed(false)
		setPreviewFailed(false)
		setPreviewIndex(0)
	}, [img, item.id, item.content_type, previewSources.length])

	function onPreviewError() {
		if (previewIndex < previewSources.length - 1) {
			setPreviewIndex(v => v + 1)
			return
		}
		setPreviewFailed(true)
	}

	return (
		<article className='card'>
			<button
				className='poster'
				onMouseEnter={() => setHovered(true)}
				onMouseLeave={() => setHovered(false)}
				onFocus={() => setHovered(true)}
				onBlur={() => setHovered(false)}
				onClick={() => onWatch(item)}
			>
				{showImage ? (
					<img
						src={img}
						alt={item.title}
						loading='lazy'
						decoding='async'
						onError={() => setImageFailed(true)}
					/>
				) : showStaticPreview ? (
					<video
						className='poster-preview-static'
						src={previewSrc}
						muted
						autoPlay
						loop
						playsInline
						preload='metadata'
						onError={onPreviewError}
					/>
				) : (
					<div className='poster-empty'>No Preview</div>
				)}
				{hovered && previewSrc && !previewFailed ? (
					<video
						className='poster-preview'
						src={previewSrc}
						muted
						autoPlay
						loop
						playsInline
						preload='metadata'
						onError={onPreviewError}
					/>
				) : null}
				<span className='play-chip'>
					<Play size={14} />
					Botda ko'rish
				</span>
			</button>
			<div className='card-body'>
				<h3>{item.title || "Noma'lum"}</h3>
				<ExpandableText
					text={item.description}
					limit={DESCRIPTION_LIMIT_CARD}
					className='card-desc'
				/>
				<div className='mini-row'>
					<span>{item.code || '-'}</span>
					<span>{item.year || '-'}</span>
					<span>{item.quality || '-'}</span>
					{genreLabel ? <span>{genreLabel}</span> : null}
				</div>
				<div className='mini-row'>
					<span>
						<ThumbsUp size={14} /> {item.likes || 0}
					</span>
					<span>
						<ThumbsDown size={14} /> {item.dislikes || 0}
					</span>
					<span>
						<MessageCircle size={14} /> {item.comments || 0}
					</span>
				</div>
				<div className='act-row'>
					<button onClick={() => onFav(item)}>
						<Bookmark size={15} /> {fav ? 'Saqlangan' : 'Saqlash'}
					</button>
					<button onClick={() => onDownload(item)}>
						<Download size={15} />
					</button>
					<button onClick={() => onShare(item)}>
						<Share2 size={15} />
					</button>
					<button onClick={() => onWatch(item)}>
						<Video size={15} />
					</button>
				</div>
			</div>
		</article>
	)
}

function HeroBanner({ item, initData, mediaToken, onWatch, onShare }) {
	const safeItem = item || {}
	const poster = resolveCardPoster(safeItem, initData, mediaToken)
	const previewSources = useMemo(
		() => resolvePreviewSources(safeItem, initData, mediaToken),
		[
			safeItem.preview_stream_file_id,
			safeItem.file_id,
			safeItem.preview_media_type,
			safeItem.media_type,
			initData,
			mediaToken,
		],
	)
	const [previewIndex, setPreviewIndex] = useState(0)
	const previewSrc = previewSources[previewIndex] || ''
	const [previewFailed, setPreviewFailed] = useState(false)
	const shortDescription = String(safeItem.description || '')
		.slice(0, 240)
		.trim()
	const genres = (safeItem.genres || []).slice(0, 3)

	useEffect(() => {
		setPreviewIndex(0)
		setPreviewFailed(false)
	}, [safeItem.id, safeItem.content_type, previewSources.length])

	function onPreviewError() {
		if (previewIndex < previewSources.length - 1) {
			setPreviewIndex(v => v + 1)
			return
		}
		setPreviewFailed(true)
	}

	if (!item) return null

	return (
		<section className='hero-banner'>
			<div className='hero-media'>
				{previewSrc && !previewFailed ? (
					<video
						src={previewSrc}
						muted
						autoPlay
						loop
						playsInline
						preload='metadata'
						poster={poster}
						onError={onPreviewError}
					/>
				) : poster ? (
					<img
						src={poster}
						alt={safeItem.title}
						loading='lazy'
						decoding='async'
					/>
				) : (
					<div className='poster-empty'>No Preview</div>
				)}
			</div>
			<div className='hero-overlay'>
				<div className='hero-content'>
					<span className='hero-kicker'>MirTopKino Premium</span>
					<h2>{safeItem.title || "Noma'lum"}</h2>
					{shortDescription ? <p>{shortDescription}</p> : null}
					<div className='hero-meta'>
						<span>{safeItem.year || '-'}</span>
						<span>{safeItem.quality || '-'}</span>
						{genres.length ? <span>{genres.join(' | ')}</span> : null}
					</div>
					<div className='hero-actions'>
						<button className='primary' onClick={() => onWatch(safeItem)}>
							<Play size={16} />
							Botda ochish
						</button>
						<button onClick={() => onShare(safeItem)}>
							<Share2 size={16} />
							Ulashish
						</button>
					</div>
				</div>
			</div>
		</section>
	)
}

function ShortCard({ item, initData, mediaToken, onWatch }) {
	const poster = resolveCardPoster(item, initData, mediaToken)
	const previewSources = useMemo(
		() => resolvePreviewSources(item, initData, mediaToken),
		[
			item.preview_stream_file_id,
			item.file_id,
			item.preview_media_type,
			item.media_type,
			initData,
			mediaToken,
		],
	)
	const [previewIndex, setPreviewIndex] = useState(0)
	const src = previewSources[previewIndex] || ''
	const [failed, setFailed] = useState(false)
	const desc = String(item.description || '').trim()

	useEffect(() => {
		setPreviewIndex(0)
		setFailed(false)
	}, [item.id, previewSources.length])

	function onMediaError() {
		if (previewIndex < previewSources.length - 1) {
			setPreviewIndex(v => v + 1)
			return
		}
		setFailed(true)
	}

	return (
		<button className='short-card' onClick={() => onWatch(item)}>
			{src && !failed ? (
				<video
					src={src}
					muted
					autoPlay
					loop
					playsInline
					preload='metadata'
					onError={onMediaError}
				/>
			) : poster ? (
				<img
					src={poster}
					alt={item.title || 'Short'}
					loading='lazy'
					decoding='async'
				/>
			) : (
				<div className='poster-empty'>Short</div>
			)}
			<div className='short-overlay'>
				<strong>{item.title || 'Short'}</strong>
				{desc ? <small>{desc.slice(0, 90)}</small> : null}
			</div>
		</button>
	)
}

function MediaShelfCard({
	item,
	initData,
	mediaToken,
	onClick,
	progressLabel = '',
	progressPercent = 0,
}) {
	const poster = resolveCardPoster(item, initData, mediaToken)
	const meta = [
		formatContentKind(item.content_type),
		item.year || '',
		item.quality || '',
	]
		.filter(Boolean)
		.join(' • ')

	return (
		<button className='media-shelf-card' onClick={onClick}>
			<div className='media-shelf-thumb'>
				{poster ? (
					<img
						src={poster}
						alt={item.title || 'Preview'}
						loading='lazy'
						decoding='async'
					/>
				) : (
					<div className='poster-empty'>Preview</div>
				)}
				<span className='media-shelf-chip'>
					<Play size={12} />
					Botda
				</span>
				{progressPercent > 0 ? (
					<span className='media-shelf-progress'>
						<span style={{ width: `${progressPercent}%` }} />
					</span>
				) : null}
			</div>
			<div className='media-shelf-body'>
				<strong>{item.title || "Noma'lum"}</strong>
				{meta ? <small>{meta}</small> : null}
				{progressLabel ? <small className='media-shelf-label'>{progressLabel}</small> : null}
			</div>
		</button>
	)
}

export default function App() {
	const initialView = useMemo(() => parseInitialViewState(), [])
	const [initData, setInitData] = useState(getTelegramInitData())
	const [theme, setTheme] = useState(
		localStorage.getItem('kino_theme') || 'dark',
	)
	const [mediaToken, setMediaToken] = useState(
		localStorage.getItem('kino_media_token') || '',
	)
	const [tab, setTab] = useState(initialView.tab)
	const [query, setQuery] = useState(initialView.query)
	const [contentType, setContentType] = useState(initialView.contentType)
	const [genreFilter, setGenreFilter] = useState(initialView.genreFilter)
	const [sortMode, setSortMode] = useState(initialView.sortMode)
	const [loading, setLoading] = useState(true)
	const [error, setError] = useState('')
	const [flash, setFlash] = useState('')
	const [boot, setBoot] = useState(null)
	const [content, setContent] = useState([])
	const [favorites, setFavorites] = useState([])
	const [history, setHistory] = useState([])
	const [trending, setTrending] = useState([])
	const [shorts, setShorts] = useState([])
	const [watch, setWatch] = useState(null)
	const [watchOpen, setWatchOpen] = useState(false)
	const [watchUrl, setWatchUrl] = useState('')
	const [watchSources, setWatchSources] = useState([])
	const [watchSourceIndex, setWatchSourceIndex] = useState(0)
	const [comments, setComments] = useState([])
	const [commentText, setCommentText] = useState('')
	const [commentSort, setCommentSort] = useState('new')
	const [replyTarget, setReplyTarget] = useState('')
	const [replyText, setReplyText] = useState('')
	const [recommendations, setRecommendations] = useState([])
	const [recommendationFeed, setRecommendationFeed] = useState({
		similar: [],
		for_you: [],
		trend: [],
		continue_watching: [],
	})
	const [adminRows, setAdminRows] = useState([])
	const [adminOverview, setAdminOverview] = useState(null)
	const [adminForm, setAdminForm] = useState({
		content_type: 'movie',
		title: '',
		code: '',
		description: '',
		year: '',
		quality: '',
		genres: '',
		file_id: '',
		preview_file_id: '',
		trailer_url: '',
		is_active: true,
		episodes_text: '',
	})
	const [notifications, setNotifications] = useState([])
	const [notifUnread, setNotifUnread] = useState(0)
	const [notifOpen, setNotifOpen] = useState(false)
	const [notifLoading, setNotifLoading] = useState(false)
	const [miniPlayer, setMiniPlayer] = useState(null)
	const [theaterMode, setTheaterMode] = useState(false)
	const [playerTime, setPlayerTime] = useState(0)
	const [playerDuration, setPlayerDuration] = useState(0)
	const [playerVolume, setPlayerVolume] = useState(1)
	const [playerRate, setPlayerRate] = useState(1)
	const [pendingStartTime, setPendingStartTime] = useState(0)
	const [gestureHint, setGestureHint] = useState('')
	const videoRef = useRef(null)
	const miniVideoRef = useRef(null)
	const playerWrapRef = useRef(null)
	const progressRef = useRef(0)
	const flashTimerRef = useRef(0)
	const touchStartXRef = useRef(null)
	const touchSeekPreviewRef = useRef(0)
	const gestureTimerRef = useRef(0)
	const blocked = Boolean(boot?.blocked)
	const isAdmin = Boolean(boot?.user?.is_admin)
	const favoriteMap = useMemo(
		() => new Set(favorites.map(x => `${x.content_type}:${x.id}`)),
		[favorites],
	)
	const continueWatching = boot?.continue_watching || []
	const profileStats = boot?.profile?.stats || boot?.stats || {}
	const homeRecommendationFeed = boot?.recommendations_feed || {
		similar: [],
		for_you: [],
		trend: [],
		continue_watching: [],
	}
	const allGenres = useMemo(() => {
		const genres = new Set()
		for (const row of content) {
			for (const value of row.genres || []) {
				const g = String(value || '').trim()
				if (g) genres.add(g)
			}
		}
		return ['all', ...Array.from(genres).sort((a, b) => a.localeCompare(b))]
	}, [content])
	const homeRows = useMemo(() => {
		const byGenre =
			genreFilter === 'all'
				? content
				: content.filter(row =>
						(row.genres || []).some(
							value =>
								String(value || '').toLowerCase() ===
								String(genreFilter || '').toLowerCase(),
						),
					)
		return sortContentRows(byGenre, sortMode)
	}, [content, genreFilter, sortMode])
	const heroItem =
		homeRecommendationFeed.for_you?.[0] || trending[0] || homeRows?.[0] || null
	const isShortForm = adminForm.content_type === 'short'
	const simplePlayer = true
	const watchPrimaryRecommendations =
		recommendationFeed.for_you?.length
			? recommendationFeed.for_you
			: recommendationFeed.similar?.length
				? recommendationFeed.similar
				: recommendations
	const watchTrendRecommendations = recommendationFeed.trend || []
	const watchContinueRecommendations = recommendationFeed.continue_watching || []
	const currentWatchKey = watch
		? `${watch.item?.content_type || ''}:${watch.item?.id || ''}`
		: ''
	const watchSidebarPrimaryRows = watchPrimaryRecommendations
		.filter(row => `${row.content_type || ''}:${row.id || ''}` !== currentWatchKey)
		.slice(0, 8)
	const watchSidebarTrendRows = watchTrendRecommendations
		.filter(row => `${row.content_type || ''}:${row.id || ''}` !== currentWatchKey)
		.slice(0, 6)
	const watchSidebarContinueRows = watchContinueRecommendations
		.filter(row => `${row.content_type || ''}:${row.id || ''}` !== currentWatchKey)
		.slice(0, 6)

	useEffect(() => {
		const tg = window.Telegram?.WebApp
		if (tg) {
			tg.ready()
			tg.expand()
			if (tg.initData) setInitData(tg.initData)
		}
	}, [])

	useEffect(() => {
		localStorage.setItem('kino_theme', theme)
		document.documentElement.setAttribute('data-theme', theme)
	}, [theme])

	useEffect(() => {
		document.title = 'MirTopKino'
	}, [])

	useEffect(() => {
		return () => {
			if (flashTimerRef.current) {
				window.clearTimeout(flashTimerRef.current)
			}
			if (gestureTimerRef.current) {
				window.clearTimeout(gestureTimerRef.current)
			}
		}
	}, [])

	useEffect(() => {
		const params = new URLSearchParams(window.location.search)
		if (query.trim()) params.set('q', query.trim())
		else params.delete('q')
		if (contentType !== 'all') params.set('type', contentType)
		else params.delete('type')
		if (tab !== T.HOME) params.set('tab', tab)
		else params.delete('tab')
		if (genreFilter !== 'all') params.set('genre', genreFilter)
		else params.delete('genre')
		if (sortMode !== 'new') params.set('sort', sortMode)
		else params.delete('sort')
		const nextQuery = params.toString()
		const nextUrl = `${window.location.pathname}${nextQuery ? `?${nextQuery}` : ''}`
		window.history.replaceState(null, '', nextUrl)
	}, [tab, query, contentType, genreFilter, sortMode])

	useEffect(() => {
		if (genreFilter !== 'all' && !allGenres.includes(genreFilter)) {
			setGenreFilter('all')
		}
	}, [genreFilter, allGenres])

	function showFlash(message) {
		setFlash(message)
		if (flashTimerRef.current) window.clearTimeout(flashTimerRef.current)
		flashTimerRef.current = window.setTimeout(() => {
			setFlash('')
		}, 2400)
	}

	function showGestureHint(text, holdMs = 700) {
		setGestureHint(text)
		if (gestureTimerRef.current) window.clearTimeout(gestureTimerRef.current)
		gestureTimerRef.current = window.setTimeout(() => {
			setGestureHint('')
		}, holdMs)
	}

	function patchRows(item, summary = {}) {
		const patch = row =>
			row.id === item.id && row.content_type === item.content_type
				? { ...row, ...summary }
				: row
		setContent(prev => prev.map(patch))
		setFavorites(prev => prev.map(patch))
		setHistory(prev => prev.map(patch))
		setTrending(prev => prev.map(patch))
		setShorts(prev => prev.map(patch))
		setBoot(prev => {
			if (!prev) return prev
			return {
				...prev,
				continue_watching: (prev.continue_watching || []).map(patch),
				shorts: (prev.shorts || []).map(patch),
				recommendations_feed: {
					...(prev.recommendations_feed || {}),
					for_you: (prev.recommendations_feed?.for_you || []).map(patch),
					trend: (prev.recommendations_feed?.trend || []).map(patch),
					similar: (prev.recommendations_feed?.similar || []).map(patch),
				},
			}
		})
	}

	function closeWatch(toMini = false) {
		const video = videoRef.current
		if (toMini && watch && watch.is_video && video && !video.paused) {
			setMiniPlayer({
				watch,
				url: watchUrl,
				sources: watchSources,
				sourceIndex: watchSourceIndex,
				episode: watch.current_episode,
				position: Math.floor(video.currentTime || 0),
				volume: Number(video.volume || 1),
				rate: Number(video.playbackRate || 1),
			})
		}
		try {
			if (document.fullscreenElement) {
				document.exitFullscreen?.()
			}
		} catch {
			// Ignore fullscreen exit errors.
		}
		try {
			window.screen?.orientation?.unlock?.()
		} catch {
			// Orientation unlock is not supported everywhere.
		}
		setWatchOpen(false)
		setWatchSources([])
		setWatchSourceIndex(0)
		setTheaterMode(false)
	}

	function applyPlayerValues() {
		const video = videoRef.current
		if (!video) return
		video.volume = playerVolume
		video.playbackRate = playerRate
	}

	async function requestPlayerFullscreen(lockLandscape = false) {
		const target = playerWrapRef.current
		if (!target) return
		try {
			if (document.fullscreenElement !== target) {
				await target.requestFullscreen?.()
			}
		} catch {
			// Fullscreen may be blocked by the environment.
		}
		if (lockLandscape) {
			try {
				await window.screen?.orientation?.lock?.('landscape')
			} catch {
				// Orientation lock may not be available.
			}
		}
	}

	async function toggleFullscreen() {
		if (!document.fullscreenElement) {
			await requestPlayerFullscreen(false)
			return
		}
		try {
			await document.exitFullscreen?.()
		} catch {
			// Ignore fullscreen exit errors.
		}
		try {
			window.screen?.orientation?.unlock?.()
		} catch {
			// Orientation unlock is not supported everywhere.
		}
	}

	async function openLandscapeMode() {
		setTheaterMode(true)
		await requestPlayerFullscreen(true)
	}

	function onWatchVideoError() {
		if (watchSourceIndex < watchSources.length - 1) {
			const nextIndex = watchSourceIndex + 1
			setWatchSourceIndex(nextIndex)
			setWatchUrl(watchSources[nextIndex] || '')
			showFlash(
				`Video manbasi almashtirildi (${nextIndex + 1}/${watchSources.length})`,
			)
			return
		}
		showFlash('Video ochilmadi. Admindan file_id ni tekshirish kerak')
	}

	function seekBy(delta) {
		const video = videoRef.current
		if (!video) return
		const next = clamp(
			Number(video.currentTime || 0) + delta,
			0,
			Number(video.duration || 0),
		)
		video.currentTime = next
		setPlayerTime(next)
		const rounded = Math.round(delta)
		if (rounded !== 0) {
			showGestureHint(`${rounded > 0 ? '+' : ''}${rounded}s`)
		}
	}

	function seekTo(value) {
		const video = videoRef.current
		if (!video) return
		const next = clamp(
			Number(value || 0),
			0,
			Number(video.duration || playerDuration || 0),
		)
		video.currentTime = next
		setPlayerTime(next)
	}

	function onTouchStart(e) {
		touchStartXRef.current = e.touches?.[0]?.clientX ?? null
		touchSeekPreviewRef.current = 0
	}

	function onTouchMove(e) {
		if (touchStartXRef.current === null) return
		const currentX = e.touches?.[0]?.clientX ?? touchStartXRef.current
		const deltaX = currentX - touchStartXRef.current
		if (Math.abs(deltaX) < 26) {
			touchSeekPreviewRef.current = 0
			return
		}
		const seconds = clamp(Math.round(Math.abs(deltaX) / 24) * 2, 2, 40)
		touchSeekPreviewRef.current = deltaX > 0 ? -seconds : seconds
		showGestureHint(
			`${touchSeekPreviewRef.current > 0 ? '+' : ''}${touchSeekPreviewRef.current}s`,
			220,
		)
	}

	function onTouchEnd(e) {
		if (touchStartXRef.current === null) return
		const endX = e.changedTouches?.[0]?.clientX ?? touchStartXRef.current
		const delta = endX - touchStartXRef.current
		touchStartXRef.current = null
		const previewSeek = Number(touchSeekPreviewRef.current || 0)
		touchSeekPreviewRef.current = 0
		if (Math.abs(delta) < 44 && !previewSeek) return
		if (previewSeek) {
			seekBy(previewSeek)
			return
		}
		if (delta > 0) seekBy(-10)
		else seekBy(10)
	}

	async function load() {
		if (!initData) {
			setLoading(false)
			setError('WebAppni bot ichidan oching yoki init_data ni yangilang.')
			return
		}
		setLoading(true)
		setError('')
		try {
			const data = await fetchBootstrap(initData)
			setBoot(data)
			const tokenValue = String(data.media_token || '').trim()
			if (tokenValue) {
				setMediaToken(tokenValue)
				localStorage.setItem('kino_media_token', tokenValue)
			}
			setContent(data.content || [])
			setFavorites(data.favorites || [])
			setHistory(data.history || [])
			setTrending(data.trending || [])
			setShorts(data.shorts || [])
			setRecommendationFeed(
				data.recommendations_feed || {
					similar: [],
					for_you: [],
					trend: [],
					continue_watching: [],
				},
			)
			setNotifications(data.notifications?.items || [])
			setNotifUnread(Number(data.notifications?.unread_count || 0))
		} catch (e) {
			setError(e.message || 'Yuklash xatosi')
		} finally {
			setLoading(false)
		}
	}

	useEffect(() => {
		load()
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [initData])

	useEffect(() => {
		if (!initData || blocked) return
		const timer = setTimeout(async () => {
			try {
				const r = await fetchContent({ initData, query, contentType })
				setContent(r.items || [])
			} catch (e) {
				setError(e.message || 'Qidiruv xatosi')
			}
		}, 260)
		return () => clearTimeout(timer)
	}, [initData, query, contentType, blocked])

	useEffect(() => {
		if (!watchOpen) return
		const handler = e => {
			const v = videoRef.current
			if (!v) return
			const key = String(e.key || '').toLowerCase()
			if (key === ' ' || key === 'k') {
				e.preventDefault()
				if (v.paused) v.play()
				else v.pause()
			} else if (key === 'arrowright' || key === 'l') {
				v.currentTime = Math.min(
					Number(v.duration || 0),
					Number(v.currentTime || 0) + 10,
				)
			} else if (key === 'arrowleft' || key === 'j') {
				v.currentTime = Math.max(0, Number(v.currentTime || 0) - 10)
			} else if (key === 'm') {
				v.muted = !v.muted
			} else if (key === 'f') {
				toggleFullscreen()
			}
		}
		window.addEventListener('keydown', handler)
		return () => window.removeEventListener('keydown', handler)
	}, [watchOpen])

	async function onFav(item) {
		await toggleFavorite({
			initData,
			contentType: item.content_type,
			contentRef: item.id,
		})
		const f = await fetchFavorites(initData)
		setFavorites(f.items || [])
		showFlash('Saqlanganlar yangilandi')
	}

	async function onDownload(item) {
		const r = await trackDownload({
			initData,
			contentType: item.content_type,
			contentRef: item.id,
		})
		patchRows(item, r.summary || {})
		if (item.deep_link) window.open(item.deep_link, '_blank')
	}

	async function onShare(item) {
		const url =
			item.deep_link ||
			`${window.location.origin}${window.location.pathname}?type=${item.content_type}`
		try {
			if (navigator.share) {
				await navigator.share({
					title: item.title || 'MirTopKino',
					text: item.description || item.title || '',
					url,
				})
				return
			}
			if (navigator.clipboard?.writeText) {
				await navigator.clipboard.writeText(url)
				showFlash('Link nusxalandi')
				return
			}
			window.prompt('Link', url)
		} catch {
			// User may cancel native share dialog.
		}
	}

	function openInBot(item) {
		const deepLink = normalizeTelegramLink(item?.deep_link)
		if (!deepLink) {
			showFlash('Bot link topilmadi')
			return
		}
		const tg = window.Telegram?.WebApp
		try {
			if (deepLink.startsWith('https://t.me/') || deepLink.startsWith('http://t.me/')) {
				tg?.openTelegramLink?.(deepLink)
				if (!tg?.openTelegramLink) {
					window.open(deepLink, '_blank', 'noopener,noreferrer')
				}
				return
			}
			tg?.openLink?.(deepLink)
			if (!tg?.openLink) {
				window.open(deepLink, '_blank', 'noopener,noreferrer')
			}
		} catch {
			window.open(deepLink, '_blank', 'noopener,noreferrer')
		}
	}

	async function loadComments(
		contentTypeValue,
		contentRefValue,
		sortValue = commentSort,
	) {
		const r = await fetchCommentsSorted({
			initData,
			contentType: contentTypeValue,
			contentRef: contentRefValue,
			sort: sortValue,
		})
		setComments(r.items || [])
	}

	async function onWatch(item, episode = null, options = {}) {
		const _ = episode
		const __ = options
		openInBot(item)
		return
		const canRecommend =
			item.content_type === 'movie' || item.content_type === 'serial'
		const recommendationTask = canRecommend
			? fetchRecommendations({
					initData,
					contentType: item.content_type,
					contentRef: item.id,
					limit: 10,
				})
			: Promise.resolve({
					items: [],
					feed: { similar: [], for_you: [], trend: [], continue_watching: [] },
				})
		const [w, rec] = await Promise.all([
			fetchWatchInfo({
				initData,
				contentType: item.content_type,
				contentRef: item.id,
				episode,
			}),
			recommendationTask,
		])
		await loadComments(item.content_type, item.id, commentSort)
		const streamFileId = String(w.stream_file_id || '').trim()
		if (!streamFileId) {
			showFlash("Bu kontentda video manzili yo'q")
			return
		}
		setWatch(w)
		const sources = buildPlayableSources(
			streamFileId,
			initData,
			mediaToken,
			String(w.media_type || ''),
		)
		setWatchSources(sources)
		setWatchSourceIndex(0)
		setWatchUrl(sources[0] || '')
		setRecommendations(rec.items || [])
		setRecommendationFeed(
			rec.feed || {
				similar: rec.items || [],
				for_you: [],
				trend: [],
				continue_watching: [],
			},
		)
		setReplyTarget('')
		setReplyText('')
		const resumeFrom =
			typeof options.startAt === 'number'
				? options.startAt
				: Number(w.playback?.position_seconds || 0)
		setPendingStartTime(Math.max(0, Math.floor(resumeFrom)))
		setPlayerTime(Math.max(0, Math.floor(resumeFrom)))
		setPlayerDuration(
			Math.max(0, Math.floor(w.playback?.duration_seconds || 0)),
		)
		setPlayerVolume(clamp(Number(options.volume ?? 1), 0, 1))
		setPlayerRate(clamp(Number(options.rate ?? 1), 0.5, 2))
		setWatchOpen(true)
		setMiniPlayer(null)
	}

	async function onProgress() {
		if (!watch?.item || !videoRef.current) return
		setPlayerTime(Number(videoRef.current.currentTime || 0))
		setPlayerDuration(Number(videoRef.current.duration || 0))
		if (Date.now() - progressRef.current < 5000) return
		progressRef.current = Date.now()
		await trackWatchProgress({
			initData,
			contentType: watch.item.content_type,
			contentRef: watch.item.id,
			episodeNumber: watch.current_episode,
			positionSeconds: Math.floor(videoRef.current.currentTime || 0),
			durationSeconds: Math.floor(videoRef.current.duration || 0),
		})
	}

	async function toggleContentReaction(nextReaction) {
		if (!watch?.item) return
		const current = String(watch.item.user_reaction || '')
		const reaction = current === nextReaction ? 'none' : nextReaction
		const r = await setReaction({
			initData,
			contentType: watch.item.content_type,
			contentRef: watch.item.id,
			reaction,
		})
		setWatch(p => ({
			...p,
			item: { ...p.item, ...(r.summary || {}) },
		}))
		patchRows(watch.item, r.summary || {})
	}

	async function onCommentReaction(commentId, current, target) {
		if (!watch?.item) return
		const reaction = current === target ? 'none' : target
		const r = await setCommentReaction({
			initData,
			commentId,
			reaction,
		})
		const summary = r.summary || {}
		setComments(prev =>
			prev.map(comment => {
				if (comment.id === commentId) return { ...comment, ...summary }
				return {
					...comment,
					replies: (comment.replies || []).map(reply =>
						reply.id === commentId ? { ...reply, ...summary } : reply,
					),
				}
			}),
		)
	}

	async function sendReply() {
		if (!watch?.item || !replyTarget || !replyText.trim()) return
		await addComment({
			initData,
			contentType: watch.item.content_type,
			contentRef: watch.item.id,
			text: replyText,
			parentCommentId: replyTarget,
		})
		setReplyText('')
		setReplyTarget('')
		await loadComments(watch.item.content_type, watch.item.id, commentSort)
	}

	async function removeComment(commentId) {
		if (!watch?.item) return
		await deleteComment({ initData, commentId })
		await loadComments(watch.item.content_type, watch.item.id, commentSort)
	}

	async function sendMainComment() {
		if (!watch?.item || !commentText.trim()) return
		await addComment({
			initData,
			contentType: watch.item.content_type,
			contentRef: watch.item.id,
			text: commentText,
		})
		setCommentText('')
		await loadComments(watch.item.content_type, watch.item.id, commentSort)
	}

	async function openMiniPlayer() {
		if (!miniPlayer) return
		await onWatch(miniPlayer.watch.item, miniPlayer.episode, {
			startAt: miniPlayer.position,
			volume: miniPlayer.volume,
			rate: miniPlayer.rate,
		})
		setMiniPlayer(null)
	}

	async function submitAdminCreate() {
		const episodes = []
		const lines = String(adminForm.episodes_text || '').split('\n')
		for (const line of lines) {
			const raw = line.trim()
			if (!raw) continue
			const parts = raw.includes('|') ? raw.split('|') : raw.split(':')
			if (parts.length < 2) continue
			const num = Number(parts[0].trim())
			const fileId = String(parts.slice(1).join(':').trim())
			if (!Number.isFinite(num) || num < 1 || !fileId) continue
			episodes.push({
				episode_number: Math.floor(num),
				file_id: fileId,
				media_type: 'video',
			})
		}
		if (
			adminForm.content_type === 'short' &&
			!String(adminForm.file_id || '').trim()
		) {
			showFlash('Short uchun video file_id majburiy')
			return
		}
		await adminCreateContent({
			initData,
			payload: {
				content_type: adminForm.content_type,
				title: adminForm.title,
				code: adminForm.code,
				description: adminForm.description,
				year: adminForm.year ? Number(adminForm.year) : null,
				quality: adminForm.quality,
				genres: String(adminForm.genres || '')
					.split(',')
					.map(x => x.trim())
					.filter(Boolean),
				file_id: adminForm.file_id,
				preview_file_id: adminForm.preview_file_id,
				preview_photo_file_id: adminForm.preview_file_id,
				preview_media_type: 'photo',
				media_type: 'video',
				trailer_url: adminForm.trailer_url,
				is_active: Boolean(adminForm.is_active),
				episodes,
			},
		})
		setAdminForm(prev => ({
			...prev,
			title: '',
			code: '',
			description: '',
			year: '',
			quality: '',
			genres: '',
			file_id: '',
			preview_file_id: '',
			trailer_url: '',
			episodes_text: '',
		}))
		showFlash("Kontent qo'shildi")
		setAdminOverview(await fetchAdminOverview(initData))
		setAdminRows((await fetchAdminContent({ initData })).items || [])
		await load()
	}

	async function syncNotifications(markAsRead = false) {
		if (!initData) return
		setNotifLoading(true)
		try {
			const data = await fetchNotifications({ initData, limit: 20 })
			setNotifications(data.items || [])
			const unread = Number(data.unread_count || 0)
			setNotifUnread(unread)
			if (markAsRead && unread > 0) {
				await markNotificationsRead(initData)
				setNotifUnread(0)
			}
		} catch (e) {
			setError(e.message || 'Bildirishnomalarni olishda xatolik')
		} finally {
			setNotifLoading(false)
		}
	}

	async function onToggleNotifications() {
		const next = !notifOpen
		setNotifOpen(next)
		if (!next) return
		await syncNotifications(true)
	}

	return (
		<main className='yt-page'>
			<header className='yt-header'>
				<div className='brand'>
					<div className='brand-dot' />
					<h1>MirTopKino</h1>
				</div>
				<div className='search'>
					<Search size={15} />
					<input
						value={query}
						onChange={e => setQuery(e.target.value)}
						placeholder='Qidiruv...'
					/>
				</div>
				<div className='tools'>
					<button className='notif-btn' onClick={onToggleNotifications}>
						<Bell size={15} />
						{notifUnread > 0 ? (
							<span className='notif-badge'>
								{notifUnread > 99 ? '99+' : notifUnread}
							</span>
						) : null}
					</button>
					<button onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
						{theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
					</button>
					<button onClick={load}>
						<Settings size={15} />
					</button>
				</div>
			</header>
			{notifOpen ? (
				<NotificationPanel
					loading={notifLoading}
					items={notifications}
					onClose={() => setNotifOpen(false)}
					onOpenItem={async item => {
						setNotifOpen(false)
						if (item?.content_type && item?.id) {
							await onWatch(item)
						}
					}}
				/>
			) : null}
			{flash ? <div className='flash'>{flash}</div> : null}

			{loading ? <SkeletonFeed /> : null}
			{error ? <section className='panel error'>{error}</section> : null}
			{!loading && blocked ? (
				<SubscriptionGate
					missingChannels={boot?.missing_channels || []}
					onRefresh={load}
				/>
			) : null}

			{!loading && !blocked ? (
				<div className='layout'>
					<aside className='nav'>
						<button
							className={tab === T.HOME ? 'active' : ''}
							onClick={() => setTab(T.HOME)}
						>
							<Home size={16} />
							Feed
						</button>
						<button
							className={tab === T.SAVED ? 'active' : ''}
							onClick={() => setTab(T.SAVED)}
						>
							<Bookmark size={16} />
							Saqlangan
						</button>
						<button
							className={tab === T.HISTORY ? 'active' : ''}
							onClick={() => setTab(T.HISTORY)}
						>
							<Clock3 size={16} />
							Tarix
						</button>
						<button
							className={tab === T.PROFILE ? 'active' : ''}
							onClick={async () => {
								setTab(T.PROFILE)
								setBoot(p => p)
								const p = await fetchProfile(initData)
								setBoot(old => ({ ...old, profile: p }))
							}}
						>
							<UserCircle2 size={16} />
							Profil
						</button>
						{isAdmin ? (
							<button
								className={tab === T.ADMIN ? 'active' : ''}
								onClick={async () => {
									setTab(T.ADMIN)
									setAdminOverview(await fetchAdminOverview(initData))
									setAdminRows(
										(await fetchAdminContent({ initData })).items || [],
									)
								}}
							>
								<Shield size={16} />
								Admin
							</button>
						) : null}
					</aside>
					<section className='main'>
						{tab === T.HOME ? (
							<>
								<HeroBanner
									item={heroItem}
									initData={initData}
									mediaToken={mediaToken}
									onWatch={onWatch}
									onShare={onShare}
								/>
								{shorts.length ? (
									<section className='panel shorts-section'>
										<h2 className='title-with-icon'>
											<Video size={16} />
											Shorts
										</h2>
										<div className='shorts-row'>
											{shorts.slice(0, 18).map(item => (
												<ShortCard
													key={`${item.content_type}:${item.id}`}
													item={item}
													initData={initData}
													mediaToken={mediaToken}
													onWatch={onWatch}
												/>
											))}
										</div>
									</section>
								) : null}
								<section className='panel'>
									<h2 className='title-with-icon'>
										<Flame size={16} />
										Trend
									</h2>
									<div className='trend-row'>
										{trending.slice(0, 10).map(x => (
											<button
												key={`${x.content_type}:${x.id}`}
												className='trend-pill'
												onClick={() => onWatch(x)}
											>
												{x.title}
											</button>
										))}
									</div>
								</section>
								{homeRecommendationFeed.for_you?.length ? (
									<section className='panel'>
										<h2 className='title-with-icon'>
											<UserCircle2 size={16} />
											Siz uchun
										</h2>
										<div className='trend-row'>
											{homeRecommendationFeed.for_you.slice(0, 12).map(x => (
												<button
													key={`${x.content_type}:${x.id}`}
													className='trend-pill'
													onClick={() => onWatch(x)}
												>
													{x.title}
												</button>
											))}
										</div>
									</section>
								) : null}
								{homeRecommendationFeed.trend?.length ? (
									<section className='panel'>
										<h2 className='title-with-icon'>
											<Video size={16} />
											Trend Pro
										</h2>
										<div className='trend-row'>
											{homeRecommendationFeed.trend.slice(0, 12).map(x => (
												<button
													key={`${x.content_type}:${x.id}`}
													className='trend-pill'
													onClick={() => onWatch(x)}
												>
													{x.title}
												</button>
											))}
										</div>
									</section>
								) : null}
								{continueWatching.length ? (
									<section className='panel'>
										<h2 className='title-with-icon'>
											<Clock3 size={16} />
											Davom Ettirish
										</h2>
										<div className='continue-row'>
											{continueWatching.slice(0, 12).map(item => {
												const progress = item.watch_progress || {}
												const position = Number(progress.position_seconds || 0)
												const duration = Number(progress.duration_seconds || 0)
												const percent =
													duration > 0
														? Math.min(
																100,
																Math.max(
																	0,
																	Math.round((position / duration) * 100),
																),
															)
														: 0
												return (
													<MediaShelfCard
														key={`${item.content_type}:${item.id}:${
															progress.episode_number || 0
														}`}
														item={item}
														initData={initData}
														mediaToken={mediaToken}
														progressPercent={percent}
														progressLabel={
															percent ? `${percent}% ko'rilgan` : 'Yangi boshlash'
														}
														onClick={() =>
															onWatch(item, progress.episode_number ?? null)
														}
													/>
												)
											})}
										</div>
									</section>
								) : null}
								<section className='panel'>
									<div className='filter filter-grid'>
										<select
											value={contentType}
											onChange={e => setContentType(e.target.value)}
										>
											<option value='all'>Barchasi</option>
											<option value='movie'>Kino</option>
											<option value='serial'>Serial</option>
											<option value='short'>Shorts</option>
										</select>
										<select
											value={sortMode}
											onChange={e => setSortMode(e.target.value)}
										>
											<option value='new'>Eng yangi</option>
											<option value='popular'>Mashhur</option>
											<option value='liked'>Like bo'yicha</option>
										</select>
										<select
											value={genreFilter}
											onChange={e => setGenreFilter(e.target.value)}
										>
											{allGenres.map(genre => (
												<option key={genre} value={genre}>
													{genre === 'all' ? 'Barcha janrlar' : genre}
												</option>
											))}
										</select>
									</div>
									<div className='grid'>
										{homeRows.length ? (
											homeRows.map(x => (
												<Card
													key={`${x.content_type}:${x.id}`}
													item={x}
													initData={initData}
													mediaToken={mediaToken}
													fav={favoriteMap.has(`${x.content_type}:${x.id}`)}
													onWatch={onWatch}
													onFav={onFav}
													onDownload={onDownload}
													onShare={onShare}
												/>
											))
										) : (
											<div className='empty-block'>Hech narsa topilmadi.</div>
										)}
									</div>
								</section>
							</>
						) : null}
						{tab === T.SAVED ? (
							<section className='panel'>
								<h2>Saqlanganlar</h2>
								<div className='grid'>
									{favorites.length ? (
										favorites.map(x => (
											<Card
												key={`${x.content_type}:${x.id}`}
												item={x}
												initData={initData}
												mediaToken={mediaToken}
												fav
												onWatch={onWatch}
												onFav={onFav}
												onDownload={onDownload}
												onShare={onShare}
											/>
										))
									) : (
										<div className='empty-block'>Saqlangan kontent yo'q.</div>
									)}
								</div>
							</section>
						) : null}
						{tab === T.HISTORY ? (
							<section className='panel'>
								<h2>Tarix</h2>
								<div className='grid'>
									{history.length ? (
										history.map(x => (
											<Card
												key={`${x.content_type}:${x.id}`}
												item={x}
												initData={initData}
												mediaToken={mediaToken}
												fav={favoriteMap.has(`${x.content_type}:${x.id}`)}
												onWatch={onWatch}
												onFav={onFav}
												onDownload={onDownload}
												onShare={onShare}
											/>
										))
									) : (
										<div className='empty-block'>Tarix hali bo'sh.</div>
									)}
								</div>
							</section>
						) : null}
						{tab === T.PROFILE ? (
							<section className='panel'>
								<h2>Profil</h2>
								<div className='profile-grid'>
									<ProfileStat
										label='Foydalanuvchi'
										value={boot?.user?.full_name || 'Foydalanuvchi'}
									/>
									<ProfileStat
										label='Telegram ID'
										value={boot?.user?.id || '-'}
									/>
									<ProfileStat
										label="Ko'rilgan"
										value={profileStats.watched ?? 0}
									/>
									<ProfileStat
										label='Saqlangan'
										value={profileStats.favorites ?? 0}
									/>
									<ProfileStat
										label='Yuklab olingan'
										value={profileStats.downloads ?? 0}
									/>
									<ProfileStat
										label='Izohlar'
										value={profileStats.comments ?? 0}
									/>
									<ProfileStat
										label='Yangi xabarlar'
										value={profileStats.notifications_unread ?? notifUnread}
									/>
								</div>
								<div className='profile-sections'>
									<div>
										<h4>Saqlangan</h4>
										<div className='profile-list'>
											{(boot?.profile?.saved || []).slice(0, 8).map(item => (
												<button
													key={`${item.content_type}:${item.id}`}
													onClick={() => onWatch(item)}
												>
													{item.title}
												</button>
											))}
										</div>
									</div>
									<div>
										<h4>Like qo'yilganlar</h4>
										<div className='profile-list'>
											{(boot?.profile?.likes || []).slice(0, 8).map(item => (
												<button
													key={`${item.content_type}:${item.id}`}
													onClick={() => onWatch(item)}
												>
													{item.title}
												</button>
											))}
										</div>
									</div>
									<div>
										<h4>Continue watching</h4>
										<div className='profile-list'>
											{(boot?.profile?.continue_watching || [])
												.slice(0, 8)
												.map(item => (
													<button
														key={`${item.content_type}:${item.id}`}
														onClick={() => onWatch(item)}
													>
														{item.title}
													</button>
												))}
										</div>
									</div>
								</div>
							</section>
						) : null}
						{tab === T.ADMIN && isAdmin ? (
							<section className='panel'>
								<h2>Admin</h2>
								<pre>{JSON.stringify(adminOverview?.stats || {}, null, 2)}</pre>
								<div className='admin-create'>
									<h3>Yangi kontent qo'shish</h3>
									<div className='admin-form-grid'>
										<select
											value={adminForm.content_type}
											onChange={e =>
												setAdminForm(prev => ({
													...prev,
													content_type: e.target.value,
												}))
											}
										>
											<option value='movie'>Movie</option>
											<option value='serial'>Serial</option>
											<option value='short'>Short</option>
										</select>
										<input
											placeholder='Title'
											value={adminForm.title}
											onChange={e =>
												setAdminForm(prev => ({
													...prev,
													title: e.target.value,
												}))
											}
										/>
										<input
											placeholder='Code'
											value={adminForm.code}
											onChange={e =>
												setAdminForm(prev => ({
													...prev,
													code: e.target.value,
												}))
											}
										/>
										{!isShortForm ? (
											<>
												<input
													placeholder='Year'
													value={adminForm.year}
													onChange={e =>
														setAdminForm(prev => ({
															...prev,
															year: e.target.value,
														}))
													}
												/>
												<input
													placeholder='Quality'
													value={adminForm.quality}
													onChange={e =>
														setAdminForm(prev => ({
															...prev,
															quality: e.target.value,
														}))
													}
												/>
												<input
													placeholder='Genres: Action,Drama'
													value={adminForm.genres}
													onChange={e =>
														setAdminForm(prev => ({
															...prev,
															genres: e.target.value,
														}))
													}
												/>
											</>
										) : null}
										<input
											placeholder='Video file_id'
											value={adminForm.file_id}
											onChange={e =>
												setAdminForm(prev => ({
													...prev,
													file_id: e.target.value,
												}))
											}
										/>
										<input
											placeholder='Poster file_id'
											value={adminForm.preview_file_id}
											onChange={e =>
												setAdminForm(prev => ({
													...prev,
													preview_file_id: e.target.value,
												}))
											}
										/>
										{!isShortForm ? (
											<input
												placeholder='Trailer URL'
												value={adminForm.trailer_url}
												onChange={e =>
													setAdminForm(prev => ({
														...prev,
														trailer_url: e.target.value,
													}))
												}
											/>
										) : null}
									</div>
									{adminForm.content_type === 'serial' ? (
										<textarea
											className='admin-episodes'
											placeholder='Serial ep: 1|file_id (har satr)'
											value={adminForm.episodes_text}
											onChange={e =>
												setAdminForm(prev => ({
													...prev,
													episodes_text: e.target.value,
												}))
											}
										/>
									) : null}
									<label className='admin-toggle'>
										<input
											type='checkbox'
											checked={adminForm.is_active}
											onChange={e =>
												setAdminForm(prev => ({
													...prev,
													is_active: e.target.checked,
												}))
											}
										/>
										<span>Publish</span>
									</label>
									<button className='primary' onClick={submitAdminCreate}>
										Qo'shish
									</button>
								</div>
								<div className='admin-list'>
									{adminRows.length ? (
										adminRows.map(x => (
											<div
												key={`${x.content_type}:${x.id}`}
												className='admin-item'
											>
												<div>
													<strong>{x.title}</strong>
													<small>
														{x.content_type} / {x.code || '-'}
														{x.content_type === 'serial'
															? ` / ${x.episodes_count || 0} ep`
															: ''}
													</small>
												</div>
												<button
													onClick={async () => {
														await adminToggleContent({
															initData,
															contentType: x.content_type,
															contentRef: x.id,
															isActive: !x.is_active,
														})
														setAdminRows(
															(await fetchAdminContent({ initData })).items ||
																[],
														)
														await load()
													}}
												>
													{x.is_active ? 'Faol' : 'Yopiq'}
												</button>
											</div>
										))
									) : (
										<div className='empty-block'>Admin kontent topilmadi.</div>
									)}
								</div>
							</section>
						) : null}
					</section>
				</div>
			) : null}

			{watchOpen && watch ? (
				<section className='player-overlay' onClick={() => closeWatch(false)}>
					<div
						className={`player-shell ${theaterMode ? 'player-shell-theater' : ''}`}
						onClick={e => e.stopPropagation()}
					>
						<header className='player-head'>
							<div className='player-heading'>
								<span className='player-kicker'>
									{formatContentKind(watch.item?.content_type)} Player
								</span>
								<h2>{watch.item?.title}</h2>
								<p>
									{[
										watch.item?.year || '',
										watch.item?.quality || '',
										(watch.item?.genres || []).slice(0, 3).join(' / '),
									]
										.filter(Boolean)
										.join(' / ') || 'Web ichida tomosha'}
								</p>
							</div>
							<div className='player-head-actions'>
								<button onClick={() => setTheaterMode(v => !v)}>
									{theaterMode ? 'Default' : 'Theater'}
								</button>
								<button onClick={openLandscapeMode}>
									<RotateCw size={15} />
									Landscape
								</button>
								<button onClick={toggleFullscreen}>
									<Fullscreen size={15} />
									Full
								</button>
								<button onClick={() => closeWatch(false)}>
									<X size={15} />
									Yopish
								</button>
							</div>
						</header>
						<div
							className={`player-grid ${simplePlayer ? 'player-grid-simple' : ''} ${
								theaterMode ? 'player-grid-theater' : ''
							}`}
						>
							<section className='watch-column'>
								{watchUrl ? (
									<div
										className='player-wrap'
										ref={playerWrapRef}
										onTouchStart={onTouchStart}
										onTouchMove={onTouchMove}
										onTouchEnd={onTouchEnd}
										onTouchCancel={onTouchEnd}
									>
										<video
											key={watchUrl}
											ref={videoRef}
											src={watchUrl}
											poster={
												watch.item?.poster_file_id || watch.item?.preview_file_id
													? buildPosterSource(
															watch.item.poster_file_id ||
																watch.item.preview_file_id,
															initData,
															mediaToken,
															watch.item.preview_media_type || 'photo',
														)
													: ''
											}
											controls
											autoPlay
											onLoadedMetadata={() => {
												applyPlayerValues()
												if (pendingStartTime > 0 && videoRef.current) {
													videoRef.current.currentTime = clamp(
														pendingStartTime,
														0,
														Number(
															videoRef.current.duration || pendingStartTime,
														),
													)
													setPendingStartTime(0)
												}
											}}
											onTimeUpdate={onProgress}
											onError={onWatchVideoError}
											playsInline
										/>
										{gestureHint ? (
											<div className='gesture-overlay'>{gestureHint}</div>
										) : null}
										<div className='player-utility-bar'>
											<div className='player-time-pill'>
												<strong>{formatDuration(playerTime)}</strong>
												<span>/ {formatDuration(playerDuration)}</span>
											</div>
											<div className='player-utility-actions'>
												<button onClick={() => seekBy(-10)}>
													<SkipBack size={15} />
													10s
												</button>
												<button
													className='primary'
													onClick={() => {
														const video = videoRef.current
														if (!video) return
														if (video.paused) video.play()
														else video.pause()
													}}
												>
													{videoRef.current?.paused ? (
														<Play size={15} />
													) : (
														<Pause size={15} />
													)}
												</button>
												<button onClick={() => seekBy(10)}>
													<Forward size={15} />
													10s
												</button>
												<label className='inline-control'>
													<Volume2 size={15} />
													<input
														type='range'
														min='0'
														max='1'
														step='0.01'
														value={playerVolume}
														onChange={e => {
															const next = Number(e.target.value || 0)
															setPlayerVolume(next)
															if (videoRef.current) videoRef.current.volume = next
														}}
													/>
												</label>
												<select
													value={playerRate}
													onChange={e => {
														const next = Number(e.target.value || 1)
														setPlayerRate(next)
														if (videoRef.current)
															videoRef.current.playbackRate = next
													}}
												>
													<option value='0.75'>0.75x</option>
													<option value='1'>1x</option>
													<option value='1.25'>1.25x</option>
													<option value='1.5'>1.5x</option>
													<option value='2'>2x</option>
												</select>
											</div>
										</div>
									</div>
								) : (
									<div className='panel'>Bu format webda ko'rsatilmaydi</div>
								)}
								<div className='watch-surface'>
									<div className='watch-title-row'>
										<div className='watch-title-copy'>
											<h3>{watch.item?.title}</h3>
											<div className='watch-stat-line'>
												<span>{watch.item?.code || 'MirTopKino'}</span>
												<span>{watch.item?.likes || 0} like</span>
												<span>{watch.item?.comments || 0} izoh</span>
												<span>{watch.item?.downloads || 0} yuklab olish</span>
											</div>
										</div>
										<div className='watch-actions'>
											<button
												className={
													watch.item?.user_reaction === 'like' ? 'active' : ''
												}
												onClick={() => toggleContentReaction('like')}
											>
												<ThumbsUp size={15} />
												{watch.item?.likes || 0}
											</button>
											<button
												className={
													watch.item?.user_reaction === 'dislike'
														? 'active'
														: ''
												}
												onClick={() => toggleContentReaction('dislike')}
											>
												<ThumbsDown size={15} />
												{watch.item?.dislikes || 0}
											</button>
											<button
												className={
													favoriteMap.has(
														`${watch.item?.content_type}:${watch.item?.id}`,
													)
														? 'active'
														: ''
												}
												onClick={() => onFav(watch.item)}
											>
												<Bookmark size={15} />
												Saqlash
											</button>
											<button onClick={() => onShare(watch.item)}>
												<Share2 size={15} />
												Ulashish
											</button>
											<button onClick={() => onDownload(watch.item)}>
												<Download size={15} />
												Yuklash
											</button>
										</div>
									</div>
									<div className='watch-chip-row'>
										<span>{formatContentKind(watch.item?.content_type)}</span>
										<span>{watch.item?.year || '-'}</span>
										<span>{watch.item?.quality || '-'}</span>
										{(watch.item?.genres || []).slice(0, 4).map(genre => (
											<span key={genre}>{genre}</span>
										))}
									</div>
									<ExpandableText
										text={watch.item?.description}
										limit={DESCRIPTION_LIMIT_WATCH}
										className='watch-desc'
									/>
								</div>
								{watch.episodes?.length ? (
									<section className='watch-section'>
										<div className='watch-section-head'>
											<h3>Epizodlar</h3>
											<small>{watch.episodes.length} ta</small>
										</div>
										<div className='episode-row'>
											{watch.episodes.map(ep => (
												<button
													key={ep.id || ep.episode_number}
													className={
														watch.current_episode === ep.episode_number
															? 'active'
															: ''
													}
													onClick={() => onWatch(watch.item, ep.episode_number)}
												>
													Ep {ep.episode_number}
												</button>
											))}
										</div>
									</section>
								) : null}
								<section className='watch-section watch-comments-panel'>
									<div className='watch-section-head'>
										<h3>Izohlar</h3>
										<select
											value={commentSort}
											onChange={async e => {
												setCommentSort(e.target.value)
												await loadComments(
													watch.item.content_type,
													watch.item.id,
													e.target.value,
												)
											}}
										>
											<option value='new'>Yangi</option>
											<option value='top'>Top</option>
											<option value='old'>Eski</option>
										</select>
									</div>
									<div className='comment-write'>
										<textarea
											value={commentText}
											onChange={e => setCommentText(e.target.value)}
											placeholder='Fikr qoldiring...'
										/>
										<button onClick={sendMainComment}>
											<Send size={14} />
										</button>
									</div>
									<div className='comment-list'>
										{comments.length ? (
											comments.map(c => (
												<div key={c.id} className='comment-thread'>
													<div className='comment-item'>
														<div className='comment-main'>
															<strong>{c.full_name}</strong>
															<small>{formatRelativeTime(c.created_at)}</small>
															<p>{c.text}</p>
															<div className='comment-actions'>
										<button
											className={
												c.user_reaction === 'like' ? 'active' : ''
											}
											onClick={() =>
												onCommentReaction(c.id, c.user_reaction, 'like')
											}
										>
											<ThumbsUp size={12} /> {c.likes || 0}
										</button>
										<button
											className={
												c.user_reaction === 'dislike' ? 'active' : ''
											}
											onClick={() =>
												onCommentReaction(c.id, c.user_reaction, 'dislike')
											}
										>
											<ThumbsDown size={12} /> {c.dislikes || 0}
										</button>
										<button
											onClick={() =>
												setReplyTarget(replyTarget === c.id ? '' : c.id)
											}
										>
											<Reply size={12} />
										</button>
										{c.can_delete ||
										Number(c.user_tg_id || 0) ===
											Number(boot?.user?.id || 0) ||
										isAdmin ? (
											<button
												onClick={() => removeComment(c.id)}
											>
												<Trash2 size={12} />
											</button>
										) : null}
															</div>
														</div>
													</div>
													{replyTarget === c.id ? (
														<div className='reply-box'>
															<textarea
																value={replyText}
																onChange={e => setReplyText(e.target.value)}
																placeholder='Javob yozing...'
															/>
															<button onClick={sendReply}>
																<Send size={12} />
															</button>
														</div>
													) : null}
													{(c.replies || []).length ? (
														<div className='reply-list'>
															{c.replies.map(reply => (
																<div
																	key={reply.id}
																	className='comment-item reply-item'
																>
																	<div className='comment-main'>
																		<strong>{reply.full_name}</strong>
																		<small>
																			{formatRelativeTime(reply.created_at)}
																		</small>
																		<p>{reply.text}</p>
																		<div className='comment-actions'>
																			<button
																				className={
																					reply.user_reaction === 'like'
																						? 'active'
																						: ''
																				}
																				onClick={() =>
																					onCommentReaction(
																						reply.id,
																						reply.user_reaction,
																						'like',
																					)
																				}
																			>
																				<ThumbsUp size={11} />{' '}
																				{reply.likes || 0}
																			</button>
																			<button
																				className={
																					reply.user_reaction === 'dislike'
																						? 'active'
																						: ''
																				}
																				onClick={() =>
																					onCommentReaction(
																						reply.id,
																						reply.user_reaction,
																						'dislike',
																					)
																				}
																			>
																				<ThumbsDown size={11} />{' '}
																				{reply.dislikes || 0}
																			</button>
																			{reply.can_delete ||
																			Number(reply.user_tg_id || 0) ===
																				Number(boot?.user?.id || 0) ||
																			isAdmin ? (
																				<button
																					onClick={() =>
																						removeComment(reply.id)
																					}
																				>
																					<Trash2 size={11} />
																				</button>
																			) : null}
																		</div>
																	</div>
																</div>
															))}
														</div>
													) : null}
												</div>
											))
										) : (
											<div className='empty-block'>
												Hozircha izohlar yo'q. Birinchi bo'lib yozing.
											</div>
										)}
									</div>
								</section>
							</section>
							<aside className='watch-sidebar'>
								{watchSidebarPrimaryRows.length ? (
									<section className='watch-side-block'>
										<div className='watch-section-head'>
											<h3>Keyingi videolar</h3>
											<small>{watchSidebarPrimaryRows.length} ta</small>
										</div>
										<div className='watch-rail-list'>
											{watchSidebarPrimaryRows.map(item => (
												<MediaShelfCard
													key={`${item.content_type}:${item.id}`}
													item={item}
													initData={initData}
													mediaToken={mediaToken}
													onClick={() => onWatch(item)}
												/>
											))}
										</div>
									</section>
								) : null}
								{watchSidebarContinueRows.length ? (
									<section className='watch-side-block'>
										<div className='watch-section-head'>
											<h3>Davom ettirish</h3>
											<small>Oxirgi ko'rilganlar</small>
										</div>
										<div className='watch-rail-list'>
											{watchSidebarContinueRows.map(item => (
												<MediaShelfCard
													key={`${item.content_type}:${item.id}`}
													item={item}
													initData={initData}
													mediaToken={mediaToken}
													onClick={() =>
														onWatch(
															item,
															item.watch_progress?.episode_number ?? null,
														)
													}
													progressPercent={Math.min(
														100,
														Math.max(
															0,
															Math.round(
																((Number(
																	item.watch_progress?.position_seconds || 0,
																) || 0) /
																	Math.max(
																		1,
																		Number(
																			item.watch_progress?.duration_seconds || 0,
																		),
																	)) *
																	100,
															),
														),
													)}
												/>
											))}
										</div>
									</section>
								) : null}
								{watchSidebarTrendRows.length ? (
									<section className='watch-side-block'>
										<div className='watch-section-head'>
											<h3>Trend</h3>
											<small>Hozir mashhur</small>
										</div>
										<div className='watch-rail-list'>
											{watchSidebarTrendRows.map(item => (
												<MediaShelfCard
													key={`${item.content_type}:${item.id}`}
													item={item}
													initData={initData}
													mediaToken={mediaToken}
													onClick={() => onWatch(item)}
												/>
											))}
										</div>
									</section>
								) : null}
							</aside>
						</div>
					</div>
				</section>
			) : null}
			{miniPlayer && !simplePlayer ? (
				<section className='mini-player'>
					<header>
						<strong>{miniPlayer.watch?.item?.title || 'Mini player'}</strong>
						<div>
							<button onClick={openMiniPlayer}>Kengaytirish</button>
							<button onClick={() => setMiniPlayer(null)}>Yopish</button>
						</div>
					</header>
					<video
						ref={miniVideoRef}
						src={miniPlayer.url}
						autoPlay
						controls
						onError={() => {
							setMiniPlayer(prev => {
								if (!prev) return prev
								const list = Array.isArray(prev.sources) ? prev.sources : []
								const idx = Number(prev.sourceIndex || 0)
								if (idx < list.length - 1) {
									const nextIndex = idx + 1
									return {
										...prev,
										sourceIndex: nextIndex,
										url: String(list[nextIndex] || prev.url),
									}
								}
								return prev
							})
						}}
						onLoadedMetadata={() => {
							if (!miniVideoRef.current) return
							miniVideoRef.current.currentTime = clamp(
								miniPlayer.position || 0,
								0,
								Number(miniVideoRef.current.duration || 0),
							)
							miniVideoRef.current.volume = clamp(miniPlayer.volume || 1, 0, 1)
							miniVideoRef.current.playbackRate = clamp(
								miniPlayer.rate || 1,
								0.5,
								2,
							)
						}}
						onTimeUpdate={() => {
							if (!miniVideoRef.current) return
							setMiniPlayer(prev =>
								prev
									? {
											...prev,
											position: Math.floor(
												miniVideoRef.current.currentTime || 0,
											),
											volume: Number(miniVideoRef.current.volume || 1),
											rate: Number(miniVideoRef.current.playbackRate || 1),
										}
									: prev,
							)
						}}
					/>
				</section>
			) : null}
		</main>
	)
}
