import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowRight, BarChart3, Bell, Bookmark, Copy, Crown, ExternalLink, Heart, Home, Megaphone, Menu, Newspaper, Play, RefreshCcw, Search, Shield, Sparkles, ThumbsDown, ThumbsUp, Trash2, User, Users, X, } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, message: '' };
    }
    static getDerivedStateFromError(error) {
        return { hasError: true, message: error?.message || 'Unknown error' };
    }
    componentDidCatch(error) {
        console.error('Miniapp render error:', error);
    }
    render() {
        if (this.state.hasError) {
            return (<div className='mesh-bg min-h-screen grid place-items-center text-white'>
					<div className='glass-panel rounded-2xl px-6 py-4 text-center text-sm'>
						UI xatoligi: {this.state.message}
					</div>
				</div>);
        }
        return this.props.children;
    }
}
const tg = window.Telegram?.WebApp ?? null;
const NAV = [
    { key: 'home', label: 'Asosiy', icon: Home },
    { key: 'search', label: 'Qidiruv', icon: Search },
    { key: 'saved', label: 'Saqlangan', icon: Bookmark },
    { key: 'pro', label: 'PRO', icon: Crown },
    { key: 'news', label: 'Yangilik', icon: Newspaper },
    { key: 'profile', label: 'Profil', icon: User },
];
const SEARCH_TYPES = [
    { key: 'all', label: 'Barchasi' },
    { key: 'movie', label: 'Kinolar' },
    { key: 'serial', label: 'Seriallar' },
];
const ADMIN_ACTIONS = [
    { action: 'open_admin_panel', label: 'Panel' },
    { action: 'admin_add_movie', label: "Kino qo'shish" },
    { action: 'admin_add_serial', label: "Serial qo'shish" },
    { action: 'admin_broadcast', label: 'Xabar' },
    { action: 'admin_stats', label: 'Statistika' },
];
function cls(...parts) {
    return parts.filter(Boolean).join(' ');
}
function compact(value) {
    const n = Number(value ?? 0);
    if (n >= 1000000)
        return `${(n / 1000000).toFixed(1)}M`;
    if (n >= 1000)
        return `${(n / 1000).toFixed(1)}K`;
    return String(n);
}
function dateText(value) {
    if (!value)
        return '-';
    const date = new Date(value);
    if (Number.isNaN(date.getTime()))
        return value;
    return date.toLocaleString('uz-UZ', {
        day: '2-digit',
        month: 'short',
        hour: '2-digit',
        minute: '2-digit',
    });
}
function statusText(value) {
    const map = {
        pending: 'Kutilmoqda',
        approved: 'Tasdiqlandi',
        rejected: 'Rad etildi',
        posted: 'Joylandi',
    };
    return map[String(value || '').toLowerCase()] || value || '-';
}
function initials(value) {
    const parts = String(value || '')
        .trim()
        .split(/\s+/)
        .filter(Boolean);
    if (!parts.length)
        return 'U';
    return parts
        .slice(0, 2)
        .map(item => item[0]?.toUpperCase() || '')
        .join('');
}
function userDisplayName(user) {
    return String(user?.full_name || user?.username || user?.id || 'User');
}
function userAvatar(user) {
    const telegramPhoto = String(tg?.initDataUnsafe?.user?.photo_url || '').trim();
    if (telegramPhoto)
        return telegramPhoto;
    const name = userDisplayName(user);
    return `https://ui-avatars.com/api/?name=${encodeURIComponent(name)}&background=0f172a&color=ffffff&bold=true`;
}
function isNewItem(item) {
    if (!item?.created_at)
        return false;
    const ts = new Date(item.created_at).getTime();
    if (Number.isNaN(ts))
        return false;
    return Date.now() - ts <= 1000 * 60 * 60 * 48;
}
function qualityScore(label) {
    const match = String(label || '').match(/(\d+)/);
    return match ? Number(match[1]) : 0;
}
function pickAutoSource(sources) {
    if (!sources?.length)
        return null;
    const sorted = [...sources].sort((a, b) => qualityScore(a.label) - qualityScore(b.label));
    const connection = navigator.connection;
    if (connection?.saveData)
        return sorted[0];
    const effective = connection?.effectiveType || '';
    if (effective === 'slow-2g' || effective === '2g')
        return sorted[0];
    if (effective === '3g')
        return sorted[Math.floor((sorted.length - 1) / 2)];
    return sorted[sorted.length - 1];
}
async function api(path, options = {}) {
    const response = await fetch(path, {
        method: options.method || 'GET',
        headers: {
            'Content-Type': 'application/json',
            'X-Telegram-Init-Data': tg?.initData || '',
        },
        body: options.body ? JSON.stringify(options.body) : undefined,
    });
    const payload = (await response
        .json()
        .catch(() => ({ ok: false, detail: 'Javob formati xato' })));
    if (!response.ok || payload.ok === false) {
        throw new Error(payload.detail || "So'rov bajarilmadi");
    }
    return payload;
}
function buildBotActionLink(action, username) {
    const user = String(username || '')
        .trim()
        .replace(/^@/, '');
    if (!user || !action)
        return '';
    return `https://t.me/${user}?start=${encodeURIComponent(`wa_${action}`)}`;
}
function openTelegramTarget(url) {
    if (!url)
        return;
    if (tg?.openTelegramLink) {
        tg.openTelegramLink(url);
        return;
    }
    if (tg?.openLink) {
        tg.openLink(url);
        return;
    }
    window.location.href = url;
}
function openUrl(url) {
    const target = String(url || '').trim();
    if (!target)
        return;
    if (target.startsWith('https://t.me/')) {
        openTelegramTarget(target);
        return;
    }
    if (tg?.openLink) {
        tg.openLink(target);
        return;
    }
    window.open(target, '_blank', 'noopener,noreferrer');
}
function Media({ item, autoPlay = false }) {
    if (item.preview_kind === 'video' && item.preview_url) {
        if (!autoPlay) {
            return (<div className='relative grid h-full w-full place-items-center bg-gradient-to-br from-accent/10 to-accent-purple/20'>
				<div className='absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.14),transparent_58%)]'/>
				<div className='glass-pill flex items-center gap-2'>
					<Play size={14}/>
					<span>Video preview</span>
				</div>
			</div>);
        }
        return (<video className='h-full w-full object-cover' src={item.preview_url} muted loop autoPlay playsInline preload='metadata'/>);
    }
    if (item.preview_url) {
        return (<img className='h-full w-full object-cover' src={item.preview_url} alt={item.title} loading='lazy' referrerPolicy='no-referrer'/>);
    }
    return (<div className='grid h-full w-full place-items-center bg-gradient-to-br from-accent/15 to-accent-purple/20 text-sm'>
			{item.code || 'MEDIA'}
		</div>);
}
function Card({ item, onOpen, onFav, onReact, }) {
    return (<article className='overflow-hidden rounded-[20px] border border-white/10 bg-white/[0.03] backdrop-blur-[18px]'>
			<button className='relative block aspect-video w-full text-left' onClick={() => onOpen(item)}>
				<Media item={item}/>
				<div className='absolute inset-0 bg-gradient-to-t from-black/65 to-transparent'/>
				<div className='absolute left-2 top-2 flex gap-1 text-[10px]'>
					<span className='glass-pill'>
						{item.content_type === 'serial' ? 'Serial' : 'Kino'}
					</span>
					<span className='glass-pill'>
						* {Number(item.rating || 0).toFixed(1)}
					</span>
                    {isNewItem(item) ? (<span className='glass-pill bg-accent/30 text-accent'>NEW</span>) : null}
                    {item.visibility && item.visibility !== 'public' ? (<span className='glass-pill bg-white/10 uppercase'>{item.visibility}</span>) : null}
				</div>
			</button>
			<div className='space-y-2 p-3'>
				<div className='flex items-center justify-between gap-2'>
					<h3 className='line-clamp-1 text-sm font-semibold'>
						{item.title || 'Nomsiz'}
					</h3>
					<span className='text-[11px] text-white/60'>{item.year || '-'}</span>
				</div>
				<p className='text-[11px] text-white/65'>
					{compact(item.views)} ko'rish - {item.quality || 'HD'} -{' '}
					{item.code || '-'}
				</p>
				<div className='flex gap-1.5'>
					<button className={cls('icon-pill', item.is_favorite && 'icon-pill-active')} onClick={() => onFav(item)}>
						<Heart size={13} fill={item.is_favorite ? 'currentColor' : 'none'}/>
					</button>
					<button className={cls('icon-pill', item.user_reaction === 'like' && 'icon-pill-active')} onClick={() => onReact(item, 'like')}>
						<ThumbsUp size={13}/>
					</button>
					<button className={cls('icon-pill', item.user_reaction === 'dislike' && 'icon-pill-active')} onClick={() => onReact(item, 'dislike')}>
						<ThumbsDown size={13}/>
					</button>
					<button className='ml-auto rounded-full bg-accent px-3 py-1 text-[11px]' onClick={() => onOpen(item)}>
						Ochish
					</button>
				</div>
			</div>
		</article>);
}
export default function App() {
    const [boot, setBoot] = useState(null);
    const [tab, setTab] = useState('home');
    const [detail, setDetail] = useState(null);
    const [playerUrl, setPlayerUrl] = useState('');
    const [playerSources, setPlayerSources] = useState([]);
    const [playerQuality, setPlayerQuality] = useState('auto');
    const [currentEpisodeIndex, setCurrentEpisodeIndex] = useState(0);
    const [toast, setToast] = useState('');
    const [loading, setLoading] = useState(true);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState('');
    const [menuOpen, setMenuOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [searchType, setSearchType] = useState('all');
    const [searchResults, setSearchResults] = useState([]);
    const [searchLoading, setSearchLoading] = useState(false);
    const [featuredIndex, setFeaturedIndex] = useState(0);
    const [adForm, setAdForm] = useState({
        title: '',
        description: '',
        buttonText: '',
        buttonUrl: '',
        photoUrl: '',
    });
    const [uploading, setUploading] = useState(false);
    const [noticeForm, setNoticeForm] = useState({ text: '', link: '' });
    const [proForm, setProForm] = useState({ priceText: '', durationDays: '' });
    const [channelInput, setChannelInput] = useState('');
    const [adminQuery, setAdminQuery] = useState('');
    const [adminUsers, setAdminUsers] = useState([]);
    const [adminSearchLoading, setAdminSearchLoading] = useState(false);
    const [adChannelMap, setAdChannelMap] = useState({});
    const toastRef = useRef();
    const videoRef = useRef(null);
    const hlsRef = useRef(null);
    const lastSaveRef = useRef(0);
    const notify = useCallback((message) => {
        setToast(message);
        if (toastRef.current)
            window.clearTimeout(toastRef.current);
        toastRef.current = window.setTimeout(() => setToast(''), 2400);
    }, []);
    const handleVideoError = useCallback(() => {
        notify("Video yuklanmadi. Bot orqali ochib ko'ring.");
    }, [notify]);
    const handlePip = useCallback(() => {
        const video = videoRef.current;
        if (!video || !document.pictureInPictureEnabled) {
            notify("PIP mavjud emas");
            return;
        }
        video.requestPictureInPicture?.().catch(() => notify("PIP ishga tushmadi"));
    }, [notify]);
    const playbackKey = useMemo(() => {
        if (!detail)
            return '';
        if (detail.episodes?.length) {
            const episode = detail.episodes[currentEpisodeIndex];
            if (episode?.episode_number)
                return `${detail.content_type}:${detail.id}:ep:${episode.episode_number}`;
        }
        return `${detail.content_type}:${detail.id}:movie`;
    }, [detail, currentEpisodeIndex]);
    const loadBoot = useCallback(async (silent = false) => {
        if (!silent)
            setLoading(true);
        setError('');
        try {
            const payload = await api('/api/bootstrap');
            try {
                localStorage.setItem('miniapp_boot', JSON.stringify(payload));
            }
            catch (_) { }
            setBoot(current => {
                const prev = current?.notice?.updated_at || '';
                const next = payload?.notice?.updated_at || '';
                if (silent && prev && next && prev !== next)
                    notify('Yangi admin xabari bor');
                return payload;
            });
        }
        catch (err) {
            setError(err instanceof Error ? err.message : 'Xatolik');
        }
        finally {
            setLoading(false);
        }
    }, [notify]);
    useEffect(() => {
        try {
            tg?.ready?.();
            tg?.expand?.();
            tg?.setHeaderColor?.('#0f1524');
            tg?.setBackgroundColor?.('#0b101b');
        }
        catch (_) { }
        let cached = null;
        try {
            cached = JSON.parse(localStorage.getItem('miniapp_boot') || 'null');
        }
        catch (_) { }
        if (cached) {
            setBoot(cached);
            setLoading(false);
        }
        void loadBoot(true);
    }, [loadBoot]);
    useEffect(() => {
        const t = window.setInterval(() => {
            if (document.visibilityState === 'visible')
                void loadBoot(true);
        }, 30000);
        return () => window.clearInterval(t);
    }, [loadBoot]);
    useEffect(() => {
        setNoticeForm({
            text: boot?.notice?.text || '',
            link: boot?.notice?.link || '',
        });
        setProForm({
            priceText: boot?.settings?.pro_price_text || '',
            durationDays: String(boot?.settings?.pro_duration_days || ''),
        });
        if (!adminQuery.trim())
            setAdminUsers(boot?.admin?.recent_users || []);
    }, [
        boot?.notice?.updated_at,
        boot?.settings?.pro_price_text,
        boot?.settings?.pro_duration_days,
        boot?.admin?.recent_users,
        adminQuery,
    ]);
    useEffect(() => {
        const first = boot?.admin?.ad_channels?.[0]?.id;
        if (!first)
            return;
        setAdChannelMap(current => {
            const next = { ...current };
            for (const ad of boot?.admin?.pending_ads || [])
                if (!next[ad.id])
                    next[ad.id] = first;
            return next;
        });
    }, [boot?.admin?.ad_channels, boot?.admin?.pending_ads]);
    const applySources = useCallback((sources) => {
        const list = Array.isArray(sources) ? sources.filter(item => item?.url) : [];
        setPlayerSources(list);
        if (!list.length) {
            setPlayerUrl('');
            setPlayerQuality('auto');
            return;
        }
        const preferred = pickAutoSource(list) || list[0];
        setPlayerUrl(preferred.url);
        setPlayerQuality(preferred.label || 'auto');
    }, []);
    const playEpisode = useCallback((episode, index) => {
        if (!episode)
            return;
        setCurrentEpisodeIndex(index);
        if (episode.media_sources?.length) {
            applySources(episode.media_sources);
            return;
        }
        if (episode.media_url) {
            applySources([{ label: 'auto', url: episode.media_url }]);
        }
    }, [applySources]);
    useEffect(() => {
        if (!detail) {
            setPlayerUrl('');
            setPlayerSources([]);
            setPlayerQuality('auto');
            setCurrentEpisodeIndex(0);
            return;
        }
        if (detail.media_sources?.length) {
            applySources(detail.media_sources);
            return;
        }
        if (detail.media_url) {
            applySources([{ label: 'auto', url: detail.media_url }]);
            return;
        }
        setCurrentEpisodeIndex(0);
        const firstEpisode = detail.episodes?.find?.(item => item.media_sources?.length || item.media_url);
        if (firstEpisode?.media_sources?.length) {
            applySources(firstEpisode.media_sources);
            return;
        }
        if (firstEpisode?.media_url) {
            applySources([{ label: 'auto', url: firstEpisode.media_url }]);
            return;
        }
        applySources([]);
    }, [applySources, detail]);
    useEffect(() => {
        const video = videoRef.current;
        if (!video)
            return;
        if (hlsRef.current) {
            hlsRef.current.destroy();
            hlsRef.current = null;
        }
        if (!playerUrl) {
            video.removeAttribute('src');
            video.load();
            return;
        }
        const isHls = playerUrl.includes('.m3u8');
        if (!isHls) {
            video.src = playerUrl;
            return;
        }
        if (video.canPlayType('application/vnd.apple.mpegurl')) {
            video.src = playerUrl;
            return;
        }
        let cancelled = false;
        import('hls.js')
            .then(module => {
            if (cancelled)
                return;
            const Hls = module.default;
            if (Hls.isSupported()) {
                const hls = new Hls({ enableWorker: true, lowLatencyMode: false });
                hls.loadSource(playerUrl);
                hls.attachMedia(video);
                hlsRef.current = hls;
            }
            else {
                video.src = playerUrl;
            }
        })
            .catch(() => {
            video.src = playerUrl;
        });
        return () => {
            cancelled = true;
            if (hlsRef.current) {
                hlsRef.current.destroy();
                hlsRef.current = null;
            }
        };
    }, [playerUrl]);
    useEffect(() => {
        const video = videoRef.current;
        if (!video || !playerUrl)
            return;
        const handleLoaded = () => {
            try {
                const raw = localStorage.getItem('miniapp_playback') || '{}';
                const saved = JSON.parse(raw);
                const time = Number(saved?.[playbackKey] || 0);
                if (time > 5 && time < video.duration) {
                    video.currentTime = time;
                }
            }
            catch (_) { }
        };
        const handleTime = () => {
            const now = Date.now();
            if (now - lastSaveRef.current < 5000)
                return;
            lastSaveRef.current = now;
            try {
                const raw = localStorage.getItem('miniapp_playback') || '{}';
                const saved = JSON.parse(raw);
                saved[playbackKey] = Math.floor(video.currentTime || 0);
                localStorage.setItem('miniapp_playback', JSON.stringify(saved));
            }
            catch (_) { }
        };
        const handleEnded = () => {
            if (!detail?.episodes?.length)
                return;
            const nextIndex = currentEpisodeIndex + 1;
            if (nextIndex >= detail.episodes.length)
                return;
            const nextEpisode = detail.episodes[nextIndex];
            playEpisode(nextEpisode, nextIndex);
        };
        video.addEventListener('loadedmetadata', handleLoaded);
        video.addEventListener('timeupdate', handleTime);
        video.addEventListener('ended', handleEnded);
        return () => {
            video.removeEventListener('loadedmetadata', handleLoaded);
            video.removeEventListener('timeupdate', handleTime);
            video.removeEventListener('ended', handleEnded);
        };
    }, [playerUrl, playbackKey, detail, currentEpisodeIndex, playEpisode]);
    const navItems = useMemo(() => {
        const items = [...NAV];
        if (boot?.user?.is_pro || boot?.user?.is_admin)
            items.splice(5, 0, {
                key: 'ads',
                label: "E'lon",
                icon: Megaphone,
            });
        if (boot?.user?.is_admin)
            items.push({ key: 'admin', label: 'Admin', icon: Shield });
        return items;
    }, [boot?.user?.is_pro, boot?.user?.is_admin]);
    const primary = useMemo(() => navItems.filter(n => ['home', 'search', 'saved', 'profile'].includes(n.key)), [navItems]);
    const secondary = useMemo(() => navItems.filter(n => !['home', 'search', 'saved', 'profile'].includes(n.key)), [navItems]);
    const secondaryActive = secondary.some(item => item.key === tab);
    const featuredItems = useMemo(() => {
        const pool = [
            ...(boot?.sections.top_viewed || []),
            ...(boot?.sections.recent_movies || []),
            ...(boot?.sections.recent_serials || []),
        ];
        const seen = new Set();
        return pool
            .filter(item => {
            const k = `${item.content_type}:${item.id}`;
            if (seen.has(k) || !item.id)
                return false;
            seen.add(k);
            return true;
        })
            .slice(0, 8);
    }, [boot?.sections]);
    useEffect(() => {
        if (featuredItems.length < 2 || tab !== 'home')
            return;
        const t = window.setInterval(() => setFeaturedIndex(c => (c + 1) % featuredItems.length), 5200);
        return () => window.clearInterval(t);
    }, [featuredItems, tab]);
    useEffect(() => {
        setMenuOpen(false);
    }, [tab]);
    const withBusy = useCallback(async (job) => {
        let active = true;
        const timer = window.setTimeout(() => {
            if (active)
                setBusy(true);
        }, 350);
        try {
            await job();
        }
        finally {
            active = false;
            window.clearTimeout(timer);
            setBusy(false);
        }
    }, []);
    const copyText = useCallback((value) => {
        navigator.clipboard
            .writeText(value)
            .then(() => notify('Nusxa olindi'))
            .catch(() => notify("Nusxa olib bo'lmadi"));
    }, [notify]);
    const sendToBot = useCallback((payload) => {
        const action = String(payload.action || '').trim();
        const deepLink = buildBotActionLink(action, boot?.links?.bot_username);
        if (deepLink)
            return openTelegramTarget(deepLink);
        if (tg?.sendData) {
            tg.sendData(JSON.stringify(payload));
            return notify("So'rov botga yuborildi");
        }
        notify('Ilovani Telegram ichida oching');
    }, [boot?.links?.bot_username, notify]);
    const openInBot = useCallback((item) => {
        if (item.deep_link)
            return openTelegramTarget(item.deep_link);
        sendToBot(item.open_payload || {});
    }, [sendToBot]);
    const openDetail = useCallback(async (item) => {
        await withBusy(async () => {
            const payload = await api(`/api/content/${item.content_type}/${item.id}`);
            setDetail(payload.item);
        }).catch((err) => notify(err instanceof Error ? err.message : 'Kontent ochilmadi'));
    }, [notify, withBusy]);
    const refreshDetail = useCallback(async (item) => {
        try {
            const payload = await api(`/api/content/${item.content_type}/${item.id}`);
            setDetail(payload.item);
        }
        catch (_) { }
    }, []);
    const toggleFavorite = useCallback(async (item) => {
        await withBusy(async () => {
            const payload = await api('/api/favorites/toggle', {
                method: 'POST',
                body: { contentType: item.content_type, contentRef: item.id },
            });
            notify(payload.active ? 'Saqlandi' : 'Saqlangandan olindi');
            await loadBoot(true);
            await refreshDetail(item);
        }).catch((err) => notify(err instanceof Error ? err.message : "Saqlab bo'lmadi"));
    }, [loadBoot, notify, refreshDetail, withBusy]);
    const reactItem = useCallback(async (item, reaction) => {
        await withBusy(async () => {
            await api('/api/reactions', {
                method: 'POST',
                body: {
                    contentType: item.content_type,
                    contentRef: item.id,
                    reaction,
                },
            });
            await loadBoot(true);
            await refreshDetail(item);
        }).catch((err) => notify(err instanceof Error ? err.message : 'Reaksiya saqlanmadi'));
    }, [loadBoot, notify, refreshDetail, withBusy]);
    const runSearch = useCallback(async () => {
        if (!searchQuery.trim())
            return setSearchResults([]);
        setSearchLoading(true);
        try {
            const payload = await api(`/api/search?q=${encodeURIComponent(searchQuery.trim())}&type=${encodeURIComponent(searchType)}`);
            setSearchResults(payload.items || []);
            notify(`${payload.items?.length || 0} ta natija`);
        }
        catch (err) {
            notify(err instanceof Error ? err.message : 'Qidiruvda xato');
        }
        finally {
            setSearchLoading(false);
        }
    }, [searchQuery, searchType, notify]);
    const toggleNotification = useCallback(async (key) => {
        await withBusy(async () => {
            await api('/api/notifications/toggle', {
                method: 'POST',
                body: { key },
            });
            await loadBoot(true);
        }).catch((err) => notify(err instanceof Error ? err.message : 'Sozlama saqlanmadi'));
    }, [loadBoot, notify, withBusy]);
    const uploadImage = useCallback(async (event) => {
        const file = event.target.files?.[0];
        if (!file)
            return;
        setUploading(true);
        try {
            const dataUrl = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(String(reader.result || ''));
                reader.onerror = () => reject(new Error("Fayl o'qilmadi"));
                reader.readAsDataURL(file);
            });
            const payload = await api('/api/upload-image', {
                method: 'POST',
                body: { dataUrl },
            });
            setAdForm(current => ({ ...current, photoUrl: payload.url }));
            notify('Rasm yuklandi');
        }
        catch (err) {
            notify(err instanceof Error ? err.message : 'Rasm yuklanmadi');
        }
        finally {
            setUploading(false);
        }
    }, [notify]);
    const submitAd = useCallback(async (event) => {
        event.preventDefault();
        await withBusy(async () => {
            await api('/api/ads', { method: 'POST', body: adForm });
            setAdForm({
                title: '',
                description: '',
                buttonText: '',
                buttonUrl: '',
                photoUrl: '',
            });
            await loadBoot(true);
            notify("E'lon moderatsiyaga yuborildi");
        }).catch((err) => notify(err instanceof Error ? err.message : "E'lon yuborilmadi"));
    }, [adForm, loadBoot, notify, withBusy]);
    const setContentMode = useCallback(async (mode) => {
        await withBusy(async () => {
            await api('/api/admin/content-mode', { method: 'POST', body: { mode } });
            await loadBoot(true);
        }).catch((err) => notify(err instanceof Error ? err.message : 'Media rejimi saqlanmadi'));
    }, [loadBoot, notify, withBusy]);
    const saveNotice = useCallback(async (clear = false) => {
        await withBusy(async () => {
            await api('/api/admin/notice', {
                method: 'POST',
                body: clear ? { text: '', link: '' } : noticeForm,
            });
            await loadBoot(true);
        }).catch((err) => notify(err instanceof Error ? err.message : 'Xabar saqlanmadi'));
    }, [loadBoot, noticeForm, notify, withBusy]);
    const saveProSettings = useCallback(async () => {
        await withBusy(async () => {
            await api('/api/admin/pro-settings', {
                method: 'POST',
                body: {
                    priceText: proForm.priceText,
                    durationDays: Number(proForm.durationDays),
                },
            });
            await loadBoot(true);
        }).catch((err) => notify(err instanceof Error ? err.message : 'PRO sozlamalari saqlanmadi'));
    }, [loadBoot, notify, proForm.durationDays, proForm.priceText, withBusy]);
    const searchAdminUsers = useCallback(async () => {
        setAdminSearchLoading(true);
        try {
            const payload = await api(`/api/admin/users/search?q=${encodeURIComponent(adminQuery.trim())}`);
            setAdminUsers(payload.items || []);
        }
        catch (err) {
            notify(err instanceof Error ? err.message : 'Foydalanuvchi qidirilmadi');
        }
        finally {
            setAdminSearchLoading(false);
        }
    }, [adminQuery, notify]);
    const setUserPro = useCallback(async (userId, enabled) => {
        await withBusy(async () => {
            const payload = await api('/api/admin/users/pro', { method: 'POST', body: { userId, enabled } });
            setAdminUsers(current => current.map(item => (item.id === userId ? payload.item : item)));
            await loadBoot(true);
        }).catch((err) => notify(err instanceof Error ? err.message : 'PRO holati yangilanmadi'));
    }, [loadBoot, notify, withBusy]);
    const setUserAdmin = useCallback(async (userId, enabled) => {
        await withBusy(async () => {
            const payload = await api('/api/admin/users/admin', { method: 'POST', body: { userId, enabled } });
            setAdminUsers(current => current.map(item => (item.id === userId ? payload.item : item)));
            await loadBoot(true);
        }).catch((err) => notify(err instanceof Error ? err.message : 'Admin holati yangilanmadi'));
    }, [loadBoot, notify, withBusy]);
    const createChannel = useCallback(async () => {
        await withBusy(async () => {
            await api('/api/admin/ad-channels/create', {
                method: 'POST',
                body: { channelRef: channelInput },
            });
            setChannelInput('');
            await loadBoot(true);
        }).catch((err) => notify(err instanceof Error ? err.message : "Kanal qo'shilmadi"));
    }, [channelInput, loadBoot, notify, withBusy]);
    const deleteChannel = useCallback(async (channelId) => {
        await withBusy(async () => {
            await api('/api/admin/ad-channels/delete', {
                method: 'POST',
                body: { channelId },
            });
            await loadBoot(true);
        }).catch((err) => notify(err instanceof Error ? err.message : "Kanal o'chirilmadi"));
    }, [loadBoot, notify, withBusy]);
    const reviewPayment = useCallback(async (requestId, action) => {
        await withBusy(async () => {
            await api('/api/admin/payments/review', {
                method: 'POST',
                body: { requestId, action },
            });
            await loadBoot(true);
        }).catch((err) => notify(err instanceof Error ? err.message : "To'lov ko'rib chiqilmadi"));
    }, [loadBoot, notify, withBusy]);
    const reviewAd = useCallback(async (adId, action) => {
        await withBusy(async () => {
            await api('/api/admin/ads/review', {
                method: 'POST',
                body: {
                    adId,
                    action,
                    channelId: action === 'approve' ? adChannelMap[adId] || '' : '',
                },
            });
            await loadBoot(true);
        }).catch((err) => notify(err instanceof Error ? err.message : "E'lon ko'rib chiqilmadi"));
    }, [adChannelMap, loadBoot, notify, withBusy]);
    const sections = boot?.sections ?? {
        recent_movies: [],
        recent_serials: [],
        top_viewed: [],
        favorites: [],
        open_requests: [],
    };
    const featured = featuredItems[featuredIndex % Math.max(featuredItems.length, 1)] || null;
    const mixedLatest = [
        ...sections.recent_movies,
        ...sections.recent_serials,
    ].slice(0, 8);
    const currentUserName = userDisplayName(boot?.user);
    const currentUserAvatar = userAvatar(boot?.user);
    const featuredHeroStyle = featured?.preview_url && featured?.preview_kind !== 'video'
        ? {
            backgroundImage: `linear-gradient(180deg, rgba(6,9,15,0.2), rgba(6,9,15,0.9)), url(${featured.preview_url})`,
        }
        : undefined;
    const news = [
        ...(boot?.notice?.text
            ? [
                {
                    id: 'notice',
                    title: 'Admin xabari',
                    description: boot.notice.text,
                    created_at: boot.notice.updated_at,
                    news_type: 'notice',
                    link: boot.notice.link,
                },
            ]
            : []),
        ...sections.recent_movies.map(item => ({
            ...item,
            news_type: 'movie',
        })),
        ...sections.recent_serials.map(item => ({
            ...item,
            news_type: 'serial',
        })),
    ].slice(0, 10);
    if (loading) {
        return (<div className='mesh-bg min-h-screen grid place-items-center text-white'>
				<div className='glass-panel rounded-2xl px-6 py-4'>Yuklanmoqda...</div>
			</div>);
    }
    if (!boot || error) {
        return (<div className='mesh-bg min-h-screen grid place-items-center text-white'>
				<div className='glass-panel rounded-2xl p-5 text-center'>
					<p className='mb-2'>Xatolik: {error || "Noma'lum"}</p>
					<button className='btn-primary' onClick={() => void loadBoot()}>
						Qayta urinish
					</button>
				</div>
			</div>);
    }
    return (<ErrorBoundary>
			<div className='mesh-bg min-h-screen text-white'>
				<div className='mx-auto max-w-[470px] px-4 pb-36 pt-4'>
				{toast ? <div className='toast-msg'>{toast}</div> : null}
				<header className='glass-panel mb-4 rounded-[22px] px-4 py-3'>
					<div className='flex items-center justify-between'>
						<div className='flex items-center gap-3'>
							<button className='icon-pill' onClick={() => setMenuOpen(true)}>
								<Menu size={15}/>
							</button>
							<div>
								<p className='text-[10px] uppercase tracking-[0.2em] text-white/65'>
									Telegram platforma
								</p>
								<p className='text-lg font-semibold'>Mir Top Kino</p>
							</div>
						</div>
						<button className='avatar-pill' onClick={() => setTab('profile')}>
							<img className='h-full w-full object-cover' src={currentUserAvatar} alt={currentUserName} referrerPolicy='no-referrer'/>
						</button>
					</div>
				</header>

				{tab === 'home' ? (<main className='space-y-4'>
						<section className='glass-panel overflow-hidden rounded-[24px]'>
							<div className='relative min-h-[280px] bg-cover bg-center' style={featuredHeroStyle}>
								{!featuredHeroStyle ? (<div className='absolute inset-0 bg-gradient-to-br from-accent-purple/35 to-accent/20'/>) : null}
								<div className='relative flex min-h-[280px] flex-col justify-end p-4'>
									<p className='text-[10px] uppercase tracking-[0.2em] text-white/70'>
										Tanlangan
									</p>
									<h2 className='mt-1 text-2xl font-semibold'>
										{featured?.title || "Kontent yo'q"}
									</h2>
									<p className='text-sm text-white/75'>
										{compact(featured?.views)} ko'rish - *{' '}
										{Number(featured?.rating || 0).toFixed(1)}
									</p>
									<div className='mt-3 flex gap-2'>
										<button className='btn-primary' onClick={() => featured ? void openDetail(featured) : setTab('search')}>
											<Play size={13}/> Davom etish
										</button>
										<button className='btn-soft' onClick={() => featured
                ? void toggleFavorite(featured)
                : setTab('saved')}>
											<Heart size={14} fill={featured?.is_favorite ? 'currentColor' : 'none'}/>
										</button>
										<button className='btn-soft' onClick={() => featured ? openInBot(featured) : undefined}>
											<ArrowRight size={14}/>
										</button>
									</div>
								</div>
							</div>
						</section>
						<section>
							<h3 className='section-title mb-2'>
								<BarChart3 size={15}/> Trend
							</h3>
							<div className='grid grid-cols-2 gap-3'>
								{boot.sections.top_viewed.map(item => (<Card key={`${item.content_type}-${item.id}`} item={item} onOpen={openDetail} onFav={toggleFavorite} onReact={reactItem}/>))}
							</div>
						</section>
						<section>
							<h3 className='section-title mb-2'>
								<Sparkles size={15}/> Yangi katalog
							</h3>
							<div className='grid grid-cols-2 gap-3'>
								{mixedLatest.map(item => (<Card key={`${item.content_type}-${item.id}`} item={item} onOpen={openDetail} onFav={toggleFavorite} onReact={reactItem}/>))}
							</div>
						</section>
					</main>) : null}

				{tab === 'search' ? (<main className='space-y-3'>
						<div className='glass-panel rounded-xl p-4'>
							<h3 className='section-title mb-2'>
								<Search size={15}/> Qidiruv
							</h3>
							<input className='input' value={searchQuery} onChange={event => setSearchQuery(event.target.value)} placeholder='Kino kodi yoki nomi'/>
							<div className='mt-2 flex gap-2'>
								{SEARCH_TYPES.map(item => (<button key={item.key} className={cls('chip', searchType === item.key && 'chip-active')} onClick={() => setSearchType(item.key)}>
										{item.label}
									</button>))}
							</div>
							<button className='btn-primary mt-2 w-full justify-center' onClick={() => void runSearch()}>
								{searchLoading ? 'Qidirilmoqda...' : 'Qidirish'}
							</button>
						</div>
						<div className='grid grid-cols-2 gap-3'>
							{searchResults.map(item => (<Card key={`${item.content_type}-${item.id}`} item={item} onOpen={openDetail} onFav={toggleFavorite} onReact={reactItem}/>))}
						</div>
					</main>) : null}
				{tab === 'saved' ? (<main>
						{boot.sections.favorites.length ? (<div className='grid grid-cols-2 gap-3'>
								{boot.sections.favorites.map(item => (<Card key={`${item.content_type}-${item.id}`} item={item} onOpen={openDetail} onFav={toggleFavorite} onReact={reactItem}/>))}
							</div>) : (<div className='glass-panel rounded-xl p-4 text-sm text-white/70'>
								Saqlanganlar bo'sh
							</div>)}
					</main>) : null}
				{tab === 'profile' ? (<main className='space-y-3'>
						<div className='glass-panel rounded-xl p-4'>
							<div className='flex items-center gap-4'>
								<div className='h-20 w-20 overflow-hidden rounded-[24px] border border-white/15 bg-white/10 shadow-[0_12px_30px_rgba(0,0,0,0.25)]'>
									<img className='h-full w-full object-cover' src={currentUserAvatar} alt={currentUserName} referrerPolicy='no-referrer'/>
								</div>
								<div className='min-w-0 flex-1'>
									<h3 className='section-title'>
										<User size={15}/> Profil
									</h3>
									<p className='mt-2 truncate text-lg font-semibold'>{currentUserName}</p>
									<p className='text-xs text-white/60'>
										{boot.user.username ? `@${boot.user.username}` : 'Telegram foydalanuvchi'}
									</p>
								</div>
							</div>
							<p className='mt-4 text-sm text-white/75'>
								ID: {boot.user.id}
								<br />
								PRO: {boot.user.is_pro ? 'Faol' : 'Faol emas'}
								<br />
								Admin: {boot.user.is_admin ? 'Ha' : "Yo'q"}
							</p>
							<div className='mt-2 flex gap-2'>
								<button className='btn-soft' onClick={() => copyText(String(boot.user.id))}>
									<Copy size={14}/> ID
								</button>
								<button className='btn-soft' onClick={() => sendToBot({ action: 'open_notifications' })}>
									<Bell size={14}/> Bot
								</button>
							</div>
						</div>
						<div className='glass-panel rounded-xl p-4 space-y-2'>
							{[
                {
                    key: 'new_content',
                    label: 'Yangi kontent',
                    value: boot.notifications.new_content,
                },
                {
                    key: 'pro_updates',
                    label: 'PRO yangiliklari',
                    value: boot.notifications.pro_updates,
                },
                {
                    key: 'ads_updates',
                    label: "E'lon xabarlari",
                    value: boot.notifications.ads_updates,
                },
            ].map(row => (<div key={row.key} className='flex items-center justify-between rounded-lg border border-white/10 bg-white/[0.02] px-3 py-2'>
									<span className='text-sm'>{row.label}</span>
									<button className={cls('chip', row.value && 'chip-active')} onClick={() => void toggleNotification(row.key)}>
										{row.value ? 'Yoqilgan' : "O'chiq"}
									</button>
								</div>))}
						</div>
					</main>) : null}
				{tab === 'news' ? (<main className='space-y-2'>
						{news.map(item => (<article key={`${item.id}-${item.created_at}`} className='glass-panel rounded-xl p-4'>
								<div className='flex items-center justify-between text-[11px] text-white/60'>
									<span>
										{item.news_type === 'notice'
                    ? 'Admin xabari'
                    : item.news_type === 'serial'
                        ? 'Yangi serial'
                        : 'Yangi kino'}
									</span>
									<span>{dateText(item.created_at)}</span>
								</div>
								<p className='mt-1 font-medium'>{item.title}</p>
								<p className='mt-1 text-sm text-white/70'>{item.description}</p>
								<div className='mt-2 flex gap-2'>
									{item.news_type === 'notice' ? (item.link ? (<button className='btn-soft' onClick={() => openUrl(item.link)}>
												Batafsil
											</button>) : null) : (<>
											<button className='btn-primary' onClick={() => void openDetail(item)}>
												Ko'rish
											</button>
											<button className='btn-soft' onClick={() => openInBot(item)}>
												Botda
											</button>
										</>)}
								</div>
							</article>))}
					</main>) : null}
				{tab === 'pro' ? (<main className='space-y-3'>
						<div className='glass-panel rounded-xl p-4'>
							<h3 className='section-title'>
								<Crown size={15}/> PRO bo'limi
							</h3>
							<p className='mt-2 text-sm text-white/75'>
								Narx: {boot.settings.pro_price_text}
								<br />
								Muddat: {boot.settings.pro_duration_days} kun
								<br />
								Holat: {boot.user.is_pro ? 'Faol' : 'Faol emas'}
								<br />
								Qachongacha: {boot.user.pro_until || '-'}
							</p>
							<div className='mt-2 flex gap-2'>
								<button className='btn-primary' onClick={() => sendToBot({ action: 'open_pro' })}>
									Davom etish
								</button>
								<button className='btn-soft' onClick={() => copyText(boot.payment.code)}>
									<Copy size={14}/>
								</button>
							</div>
						</div>
						<div className='glass-panel rounded-xl p-4'>
							<p className='text-xs text-white/60'>To'lov kodi</p>
							<p className='text-2xl font-semibold'>{boot.payment.code}</p>
							<div className='mt-2 flex gap-2'>
								{boot.payment.links.map((link, index) => (<button key={link} className='btn-soft' onClick={() => openUrl(link)}>
										<ExternalLink size={14}/> To'lov {index + 1}
									</button>))}
							</div>
						</div>
					</main>) : null}
				{tab === 'ads' ? (<main className='space-y-3'>
						<div className='glass-panel rounded-xl p-4'>
							{boot.ads.mine.length ? (<div className='space-y-2'>
									{boot.ads.mine.map(item => (<div key={item.id} className='rounded-lg border border-white/10 bg-white/[0.02] p-3'>
											<p className='font-medium'>{item.title}</p>
											<p className='text-xs text-white/60'>
												{statusText(item.status)} - {dateText(item.created_at)}
											</p>
											<p className='mt-1 text-sm text-white/70'>
												{item.description}
											</p>
										</div>))}
								</div>) : (<p className='text-sm text-white/70'>E'lonlar yo'q</p>)}
						</div>
						<form className='glass-panel rounded-xl p-4 space-y-2' onSubmit={event => void submitAd(event)}>
							{boot.ads.can_create ? (<>
									<input className='input' value={adForm.title} onChange={event => setAdForm(c => ({ ...c, title: event.target.value }))} placeholder='Sarlavha'/>
									<textarea className='input min-h-[90px]' value={adForm.description} onChange={event => setAdForm(c => ({
                    ...c,
                    description: event.target.value,
                }))} placeholder='Tavsif'/>
									<input className='input' value={adForm.buttonText} onChange={event => setAdForm(c => ({ ...c, buttonText: event.target.value }))} placeholder='Tugma matni'/>
									<input className='input' value={adForm.buttonUrl} onChange={event => setAdForm(c => ({ ...c, buttonUrl: event.target.value }))} placeholder='https://...'/>
									<input className='input' type='file' accept='image/*' onChange={event => void uploadImage(event)}/>
									<p className='text-xs text-white/60'>
										{uploading
                    ? 'Yuklanmoqda...'
                    : adForm.photoUrl
                        ? 'Rasm tayyor'
                        : 'Rasm ixtiyoriy'}
									</p>
									<button className='btn-primary' type='submit'>
										Yuborish
									</button>
								</>) : (<p className='text-sm text-white/70'>E'lon uchun PRO kerak</p>)}
						</form>
					</main>) : null}
				{tab === 'admin' && boot.user.is_admin ? (<main className='space-y-3'>
						<div className='grid grid-cols-2 gap-2'>
							{[
                {
                    label: 'Foydalanuvchilar',
                    value: boot.admin?.total_users || 0,
                    icon: Users,
                },
                {
                    label: 'PRO',
                    value: boot.admin?.total_pro_users || 0,
                    icon: Crown,
                },
                {
                    label: 'Kinolar',
                    value: boot.admin?.total_movies || 0,
                    icon: Play,
                },
                {
                    label: 'Seriallar',
                    value: boot.admin?.total_serials || 0,
                    icon: Sparkles,
                },
            ].map(item => (<div key={item.label} className='glass-panel rounded-xl p-3'>
									<item.icon size={14} className='text-accent'/>
									<p className='text-[11px] text-white/60'>{item.label}</p>
									<p className='text-lg font-semibold'>{item.value}</p>
								</div>))}
						</div>
						<div className='glass-panel rounded-xl p-4'>
							<h3 className='section-title'>
								<Shield size={15}/> Tezkor amallar
							</h3>
							<div className='mt-2 grid grid-cols-2 gap-2'>
								{ADMIN_ACTIONS.map(item => (<button key={item.action} className='chip justify-center' onClick={() => sendToBot({ action: item.action })}>
										{item.label}
									</button>))}
							</div>
						</div>
						<div className='glass-panel rounded-xl p-4'>
							<div className='flex gap-2'>
								<button className='btn-soft' onClick={() => void setContentMode('private')}>
									Yopiq
								</button>
								<button className='btn-soft' onClick={() => void setContentMode('public')}>
									Ochiq
								</button>
							</div>
						</div>
						<div className='glass-panel rounded-xl p-4'>
							<h3 className='section-title'>
								<Users size={15}/> Foydalanuvchilar
							</h3>
							<div className='mt-2 flex gap-2'>
								<input className='input' value={adminQuery} onChange={event => setAdminQuery(event.target.value)} placeholder='ID yoki ism'/>
								<button className='btn-soft' onClick={() => void searchAdminUsers()} disabled={adminSearchLoading}>
									<Search size={14}/>
								</button>
							</div>
							<div className='mt-2 space-y-2'>
								{adminUsers.length ? (adminUsers.map(item => (<div key={item.id} className='rounded-lg border border-white/10 bg-white/[0.02] p-3'>
											<p className='font-medium'>
												{item.full_name || `User ${item.id}`}
											</p>
											<p className='text-xs text-white/60'>
												ID {item.id} - {dateText(item.joined_at)}
											</p>
											<div className='mt-2 flex gap-2'>
												<button className='chip' onClick={() => void setUserPro(item.id, !item.is_pro)}>
													PRO {item.is_pro ? "o'chirish" : 'yoqish'}
												</button>
												<button className='chip' disabled={item.is_seed_admin ||
                    (item.id === boot.user.id && item.is_admin)} onClick={() => void setUserAdmin(item.id, !item.is_admin)}>
													Admin {item.is_admin ? 'olish' : 'berish'}
												</button>
											</div>
										</div>))) : (<p className='text-sm text-white/65'>Foydalanuvchi yo'q</p>)}
							</div>
						</div>
						<div className='glass-panel rounded-xl p-4'>
							<h3 className='section-title'>
								<Megaphone size={15}/> Sayt xabari
							</h3>
							<textarea className='input mt-2 min-h-[90px]' value={noticeForm.text} onChange={event => setNoticeForm(c => ({ ...c, text: event.target.value }))} placeholder='Xabar matni'/>
							<input className='input mt-2' value={noticeForm.link} onChange={event => setNoticeForm(c => ({ ...c, link: event.target.value }))} placeholder='Havola'/>
							<div className='mt-2 flex gap-2'>
								<button className='btn-primary' onClick={() => void saveNotice(false)}>
									Saqlash
								</button>
								<button className='btn-soft' onClick={() => void saveNotice(true)}>
									Tozalash
								</button>
							</div>
						</div>
						<div className='glass-panel rounded-xl p-4'>
							<h3 className='section-title'>
								<Crown size={15}/> PRO sozlamalari
							</h3>
							<input className='input mt-2' value={proForm.priceText} onChange={event => setProForm(c => ({ ...c, priceText: event.target.value }))} placeholder='Narx matni'/>
							<input className='input mt-2' type='number' value={proForm.durationDays} onChange={event => setProForm(c => ({ ...c, durationDays: event.target.value }))} placeholder='Muddat kun'/>
							<button className='btn-primary mt-2' onClick={() => void saveProSettings()}>
								Saqlash
							</button>
						</div>
						<div className='glass-panel rounded-xl p-4'>
							<h3 className='section-title'>
								<Megaphone size={15}/> E'lon kanallari
							</h3>
							<div className='mt-2 flex gap-2'>
								<input className='input' value={channelInput} onChange={event => setChannelInput(event.target.value)} placeholder='@kanal yoki -100...'/>
								<button className='btn-soft' onClick={() => void createChannel()}>
									<ArrowRight size={14}/>
								</button>
							</div>
							<div className='mt-2 space-y-2'>
								{(boot.admin?.ad_channels || []).map(channel => (<div key={channel.id} className='flex items-center justify-between rounded-lg border border-white/10 bg-white/[0.02] px-3 py-2'>
										<div>
											<p>{channel.title}</p>
											<p className='text-xs text-white/60'>
												{channel.channel_ref}
											</p>
										</div>
										<button className='icon-pill' onClick={() => void deleteChannel(channel.id)}>
											<Trash2 size={13}/>
										</button>
									</div>))}
							</div>
						</div>
						<div className='glass-panel rounded-xl p-4'>
							<h3 className='section-title'>
								<Bell size={15}/> Kutilayotgan to'lovlar
							</h3>
							<div className='mt-2 space-y-2'>
								{(boot.admin?.pending_payments || []).length ? (boot.admin?.pending_payments.map(item => (<div key={item.id} className='rounded-lg border border-white/10 bg-white/[0.02] p-3'>
											<p className='font-medium'>User {item.user_tg_id}</p>
											<p className='text-xs text-white/60'>
												{item.payment_code} - {dateText(item.created_at)}
											</p>
											<div className='mt-2 flex gap-2'>
												<button className='chip chip-approve' onClick={() => void reviewPayment(item.id, 'approve')}>
													Tasdiqlash
												</button>
												<button className='chip chip-reject' onClick={() => void reviewPayment(item.id, 'reject')}>
													Rad etish
												</button>
											</div>
										</div>))) : (<p className='text-sm text-white/65'>To'lov yo'q</p>)}
							</div>
						</div>
						<div className='glass-panel rounded-xl p-4'>
							<h3 className='section-title'>
								<Megaphone size={15}/> Kutilayotgan e'lonlar
							</h3>
							<div className='mt-2 space-y-2'>
								{(boot.admin?.pending_ads || []).length ? (boot.admin?.pending_ads.map(item => (<div key={item.id} className='rounded-lg border border-white/10 bg-white/[0.02] p-3'>
											<p className='font-medium'>{item.title}</p>
											<p className='text-xs text-white/60'>
												User {item.user_tg_id} - {dateText(item.created_at)}
											</p>
											<p className='mt-1 text-sm text-white/70'>
												{item.description}
											</p>
											<select className='input mt-2' value={adChannelMap[item.id] || ''} onChange={event => setAdChannelMap(c => ({
                    ...c,
                    [item.id]: event.target.value,
                }))}>
												{(boot.admin?.ad_channels || []).map(channel => (<option key={channel.id} value={channel.id}>
														{channel.title}
													</option>))}
											</select>
											<div className='mt-2 flex gap-2'>
												<button className='chip chip-approve' onClick={() => void reviewAd(item.id, 'approve')}>
													Tasdiqlash
												</button>
												<button className='chip chip-reject' onClick={() => void reviewAd(item.id, 'reject')}>
													Rad etish
												</button>
											</div>
										</div>))) : (<p className='text-sm text-white/65'>E'lon yo'q</p>)}
							</div>
						</div>
					</main>) : null}
				<AnimatePresence>
					{detail ? (<motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className='fixed inset-0 z-[90] bg-black/50 backdrop-blur-sm' onClick={() => setDetail(null)}>
							<motion.div initial={{ y: '100%' }} animate={{ y: 0 }} exit={{ y: '100%' }} transition={{ type: 'spring', damping: 30, stiffness: 320 }} className='fixed bottom-0 left-1/2 h-[88vh] w-full max-w-[470px] -translate-x-1/2 overflow-hidden rounded-t-[28px] border border-white/10 bg-[#121724]/95' onClick={event => event.stopPropagation()}>
								<button className='icon-pill absolute right-4 top-4 z-10' onClick={() => setDetail(null)}>
									<X size={14}/>
								</button>
                                <div className='h-[42vh] bg-black/40'>
                                    {playerUrl ? (<video key={playerUrl} ref={videoRef} className='h-full w-full object-cover' src={playerUrl} poster={detail.preview_url || ''} controls playsInline preload='metadata' onError={handleVideoError}/>) : (<Media item={detail} autoPlay={true}/>)}
                                </div>
								<div className='space-y-3 p-4'>
									<h3 className='text-2xl font-semibold'>{detail.title}</h3>
									<p className='text-sm text-white/75'>
										{detail.description || "Tavsif yo'q"}
									</p>
									<div className='grid grid-cols-2 gap-2 text-sm'>
										<div className='glass-panel rounded-lg p-2'>
											Korishlar: {compact(detail.views)}
										</div>
										<div className='glass-panel rounded-lg p-2'>
											Yuklash: {compact(detail.downloads)}
										</div>
										<div className='glass-panel rounded-lg p-2'>
											Yoqdi: {compact(detail.likes)}
										</div>
										<div className='glass-panel rounded-lg p-2'>
											Yoqmadi: {compact(detail.dislikes)}
										</div>
									</div>
                                    <div className='flex gap-2'>
                                        <button className='btn-primary' onClick={() => openInBot(detail)}>
                                            <Play size={14}/> Botda davom etish
                                        </button>
										<button className='btn-soft' onClick={() => void toggleFavorite(detail)}>
											<Heart size={14} fill={detail.is_favorite ? 'currentColor' : 'none'}/>
										</button>
                                        <button className='btn-soft' onClick={() => copyText(detail.deep_link || detail.code || detail.title)}>
                                            <ExternalLink size={14}/>
                                        </button>
                                        {playerUrl ? (<button className='btn-soft' onClick={handlePip}>
                                                <Sparkles size={14}/>
                                            </button>) : null}
                                    </div>
                                    {playerSources.length ? (<div className='mt-2'>
                                            <p className='mb-2 text-xs uppercase tracking-[0.2em] text-white/60'>Sifat</p>
                                            <div className='flex flex-wrap gap-2'>
                                                {playerSources.map(source => (<button key={`${source.label}-${source.url}`} className={cls('chip', source.url === playerUrl && 'chip-active')} onClick={() => {
                                                        setPlayerUrl(source.url);
                                                        setPlayerQuality(source.label || 'auto');
                                                    }}>
                                                        {source.label || 'auto'}
                                                    </button>))}
                                            </div>
                                        </div>) : null}
                                    {detail.episodes?.length ? (<div className='mt-2'>
                                            <p className='mb-2 text-xs uppercase tracking-[0.2em] text-white/60'>Qismlar</p>
                                            <div className='flex flex-wrap gap-2'>
                                                {detail.episodes.map((item, index) => (<button key={item.episode_number} className={cls('chip', index === currentEpisodeIndex && 'chip-active')} disabled={!item.media_url && !item.media_sources?.length} onClick={() => playEpisode(item, index)}>
                                                        {item.episode_number}
                                                    </button>))}
                                            </div>
                                        </div>) : null}
									<div className='flex gap-2'>
										<button className={cls('btn-soft flex-1', detail.user_reaction === 'like' && 'btn-soft-active')} onClick={() => void reactItem(detail, 'like')}>
											<ThumbsUp size={14}/> {compact(detail.likes)}
										</button>
										<button className={cls('btn-soft flex-1', detail.user_reaction === 'dislike' &&
                'btn-soft-active')} onClick={() => void reactItem(detail, 'dislike')}>
											<ThumbsDown size={14}/> {compact(detail.dislikes)}
										</button>
									</div>
								</div>
							</motion.div>
						</motion.div>) : null}
				</AnimatePresence>
			</div>
			<nav className='fixed bottom-4 left-1/2 z-40 w-[calc(100%-16px)] max-w-[470px] -translate-x-1/2 px-2'>
				<div className='glass-panel rounded-[26px] px-2 py-2'>
					<div className='flex items-center justify-around gap-1'>
						{primary.map(item => {
            const Icon = item.icon;
            return (<button key={item.key} className={cls('dock-item', tab === item.key && 'dock-item-active')} onClick={() => setTab(item.key)}>
									<Icon size={16}/>
									<span>{item.label}</span>
								</button>);
        })}
					</div>
				</div>
			</nav>
			<AnimatePresence>
				{menuOpen ? (<motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className='fixed inset-0 z-[80] bg-black/40 backdrop-blur-sm' onClick={() => setMenuOpen(false)}>
						<motion.div initial={{ y: 40, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: 40, opacity: 0 }} className='absolute bottom-24 left-1/2 w-[calc(100%-28px)] max-w-[430px] -translate-x-1/2 rounded-[24px] border border-white/10 bg-[#141a28]/95 p-3' onClick={event => event.stopPropagation()}>
							<div className='mb-2 flex items-center justify-between'>
								<p className='text-xs uppercase tracking-[0.2em] text-white/60'>
									Bolimlar
								</p>
								<button className='icon-pill' onClick={() => setMenuOpen(false)}>
									<X size={13}/>
								</button>
							</div>
							<div className='grid grid-cols-2 gap-2'>
								{secondary.map(item => {
                const Icon = item.icon;
                return (<button key={item.key} className={cls('menu-card', tab === item.key && 'menu-card-active')} onClick={() => {
                        setTab(item.key);
                        setMenuOpen(false);
                    }}>
											<Icon size={15}/>
											<span>{item.label}</span>
										</button>);
            })}
							</div>
						</motion.div>
					</motion.div>) : null}
			</AnimatePresence>
			<AnimatePresence>
				{busy ? (<motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className='fixed inset-0 z-[100] grid place-items-center bg-black/20 backdrop-blur-sm'>
						<div className='glass-panel rounded-xl px-5 py-4 text-center'>
							<RefreshCcw size={16} className='mx-auto animate-spin text-accent'/>
							<p className='mt-2 text-xs uppercase tracking-[0.2em] text-white/70'>
								Yuklanmoqda
							</p>
						</div>
					</motion.div>) : null}
			</AnimatePresence>
		</div>
		</ErrorBoundary>);
}


