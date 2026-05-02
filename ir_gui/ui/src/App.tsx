import { useState, useEffect, useRef } from 'react'

const API_URL = 'http://localhost:8000'
const WS_URL = 'ws://localhost:8000/ws/video'
type AppMode = 'single' | 'paired' | 'grayscale'

const TRUST_COLORS: Record<number, string> = {
    0: 'text-red-400',      // reject_both
    1: 'text-blue-400',     // trust_rgb
    2: 'text-orange-400',   // trust_ir
    3: 'text-emerald-400',  // trust_both
}
const TRUST_BG: Record<number, string> = {
    0: 'bg-red-500/20 border-red-500/50',
    1: 'bg-blue-500/20 border-blue-500/50',
    2: 'bg-orange-500/20 border-orange-500/50',
    3: 'bg-emerald-500/20 border-emerald-500/50',
}

interface VerifierStats {
    active: boolean
    vetoed: boolean
    threshold: number
    original_trust: number
    original_trust_name: string
    rgb_max_p: number | null
    ir_max_p: number | null
    rgb_n_boxes: number
    ir_n_boxes: number
    rgb_skipped_reason: string | null
    ir_skipped_reason: string | null
    rgb_labels: string[]
    ir_labels: string[]
}

interface FusionStats {
    frame_id: number
    total_frames: number
    video_fps: number
    trust_label: number
    trust_name: string
    trust_probs: number[]
    rgb_dets: number
    ir_dets: number
    infer_ms: number
    mode: string
    playback_speed: number
    warning_active: boolean
    alert_active: boolean
    warning_events: number
    alert_events: number
    verifier?: VerifierStats | null
}

// SVG Icons
const DroneIcon = () => (
    <svg className="w-6 h-6" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2L4 5v6.09c0 5.05 3.41 9.76 8 10.91 4.59-1.15 8-5.86 8-10.91V5l-8-3zm-1.06 13.54L7.4 12l1.41-1.41 2.12 2.12 4.24-4.24 1.41 1.41-5.64 5.66z" />
    </svg>
)

const CameraIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
    </svg>
)

const FolderIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
    </svg>
)

const PlayIcon = () => (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M8 5v14l11-7z" />
    </svg>
)

const PauseIcon = () => (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
    </svg>
)

const StopIcon = () => (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M6 6h12v12H6z" />
    </svg>
)





const AlertIcon = () => (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
    </svg>
)

const DownloadIcon = () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
    </svg>
)

const WifiIcon = () => (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M1 9l2 2c4.97-4.97 13.03-4.97 18 0l2-2C16.93 2.93 7.08 2.93 1 9zm8 8l3 3 3-3c-1.65-1.66-4.34-1.66-6 0zm-4-4l2 2c2.76-2.76 7.24-2.76 10 0l2-2C15.14 9.14 8.87 9.14 5 13z" />
    </svg>
)

const MaximizeIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 4l-5-5M4 16v4m0 0h4M4 20l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
    </svg>
)

const MinimizeIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 9L4 4m0 0l5 5M4 4h4.5M4 4v4.5M15 9l5-5m0 0l-5 5M20 4v4.5M20 4h-4.5M9 15l-5 5m0 0l5-5M4 20v-4.5M4 20h4.5M15 15l5 5m0 0l-5-5M20 20h-4.5M20 20v-4.5" />
    </svg>
)

const ThemeIcon = ({ theme }: { theme: 'dark' | 'light' }) => (
    theme === 'dark'
        ? <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
        </svg>
        : <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
        </svg>
)

const SettingsIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
)

interface Stats {
    fps: number
    frame: number
    detections: number
    alerts: number
    warnings: number
    // Window progress (for progress bars)
    warn_hits: number
    warn_need: number
    alert_hits: number
    alert_need: number
    // Video info
    total_frames: number
    video_fps: number
    playback_speed: number
}

interface Settings {
    cascade_mode: string
    temporal_roi_enabled: boolean
    infer_fps: number
    show_gate: boolean
    show_troi: boolean
    show_cascade: boolean
    log_mode: string
    save_video: boolean
    save_alert_frames: boolean
    warning_cooldown: number
    alert_cooldown: number
    detect_conf: number
    cascade_trigger_conf: number
    cascade_accept_conf: number
    roi_size: number
    // Window settings
    warning_window_size: number
    warning_require_hits: number
    alert_window_size: number
    alert_require_hits: number
    alert_min_area: number
    simple_mode: boolean
    alert_avg_conf_threshold: number
    // Fusion confidence thresholds
    rgb_conf: number
    ir_conf: number
    ir_conf_grayscale: number
    // Patch verifier
    use_patch_verifier: boolean
    patch_threshold: number
}

export default function App() {
    const [connected, setConnected] = useState(false)
    const [streaming, setStreaming] = useState(false)
    const [paused, setPaused] = useState(false)
    const [isFullscreen, setIsFullscreen] = useState(false)
    const [alertActive, setAlertActive] = useState(false)
    const [warningActive, setWarningActive] = useState(false)
    const [stats, setStats] = useState<Stats>({
        fps: 0, frame: 0, detections: 0, alerts: 0, warnings: 0,
        warn_hits: 0, warn_need: 9, alert_hits: 0, alert_need: 9,
        total_frames: 0, video_fps: 0, playback_speed: 1.0
    })
    const [playbackSpeed, setPlaybackSpeed] = useState(1.0)
    const [source, setSource] = useState('')
    const [logs, setLogs] = useState<string[]>(() => {
        const saved = localStorage.getItem('droneDetectionLogs')
        return saved ? JSON.parse(saved) : []
    })
    const [showSettings, setShowSettings] = useState(false)
    const [pendingSettings, setPendingSettings] = useState<Settings | null>(null)
    const [settings, setSettings] = useState<Settings>({
        cascade_mode: 'None',
        temporal_roi_enabled: true,
        infer_fps: 5,
        show_gate: false,
        show_troi: false,
        show_cascade: false,
        log_mode: 'windows_big',
        save_video: false,
        save_alert_frames: true,
        warning_cooldown: 3.0,
        alert_cooldown: 3.0,
        detect_conf: 0.25,
        cascade_trigger_conf: 0.40,
        cascade_accept_conf: 0.40,
        roi_size: 640,
        warning_window_size: 10,
        warning_require_hits: 9,
        alert_window_size: 10,
        alert_require_hits: 9,
        alert_min_area: 625,
        simple_mode: false,
        alert_avg_conf_threshold: 0.0,
        rgb_conf: 0.25,
        ir_conf: 0.40,
        ir_conf_grayscale: 0.05,
        use_patch_verifier: true,
        patch_threshold: 0.70,
    })

    // Fetch settings on mount
    useEffect(() => {
        fetch(`${API_URL}/api/settings`)
            .then(res => res.json())
            .then(data => setSettings(prev => ({ ...prev, ...data })))
            .catch(() => addLog('Failed to load settings'))
    }, [])

    useEffect(() => {
        localStorage.setItem('droneDetectionLogs', JSON.stringify(logs))
    }, [logs])

    const [youtubeUrl, setYoutubeUrl] = useState('')
    const [baseDir, setBaseDir] = useState(() => localStorage.getItem('droneBaseDir') || '')
    const [selectedFile, setSelectedFile] = useState('')
    const [esp32Addr, setEsp32Addr] = useState('172.20.10.9:5000')
    const [streamQuality, setStreamQuality] = useState('1080p')
    const [device, setDevice] = useState('gpu')

    // App mode
    const [appMode, setAppMode] = useState<AppMode>('single')

    // Fusion state
    const [fusionStreaming, setFusionStreaming] = useState(false)
    const [fusionSessionId, setFusionSessionId] = useState(0)  // cache-bust MJPEG URL
    const [fusionPaused, setFusionPaused] = useState(false)
    const [fusionStats, setFusionStats] = useState<FusionStats | null>(null)
    const [fusionRgbPath, setFusionRgbPath] = useState(() => localStorage.getItem('fusionRgbPath') || '')
    const [fusionIrPath, setFusionIrPath] = useState(() => localStorage.getItem('fusionIrPath') || '')
    const [fusionBaseDir, setFusionBaseDir] = useState(() => localStorage.getItem('fusionBaseDir') || '')
    const [fusionYoutubeUrl, setFusionYoutubeUrl] = useState('')
    const [fusionSpeed, setFusionSpeed] = useState(1)
    const [fusionConnected, setFusionConnected] = useState(false)

    // Model selection
    const [modelBaseDir, setModelBaseDir] = useState(() => localStorage.getItem('droneModelBaseDir') || '')
    const [modelFile, setModelFile] = useState('')
    const [modelLoaded, setModelLoaded] = useState(false)
    const modelInputRef = useRef<HTMLInputElement>(null)

    const imgRef = useRef<HTMLImageElement>(null)
    const fileInputRef = useRef<HTMLInputElement>(null)
    const fusionRgbInputRef = useRef<HTMLInputElement>(null)
    const fusionIrInputRef = useRef<HTMLInputElement>(null)
    const wsRef = useRef<WebSocket | null>(null)

    useEffect(() => {
        localStorage.setItem('droneBaseDir', baseDir)
    }, [baseDir])

    useEffect(() => {
        localStorage.setItem('droneModelBaseDir', modelBaseDir)
    }, [modelBaseDir])

    useEffect(() => {
        localStorage.setItem('fusionBaseDir', fusionBaseDir)
    }, [fusionBaseDir])

    useEffect(() => {
        localStorage.setItem('fusionRgbPath', fusionRgbPath)
    }, [fusionRgbPath])

    useEffect(() => {
        localStorage.setItem('fusionIrPath', fusionIrPath)
    }, [fusionIrPath])

    const fullPath = baseDir && selectedFile ?
        `${baseDir}${baseDir.endsWith('\\') || baseDir.endsWith('/') ? '' : '\\'}${selectedFile}` : ''

    const fullModelPath = modelBaseDir && modelFile ?
        `${modelBaseDir}${modelBaseDir.endsWith('\\') || modelBaseDir.endsWith('/') ? '' : '\\'}${modelFile}` : ''

    // Load settings + model on mount
    useEffect(() => {
        fetch(`${API_URL}/api/settings`)
            .then(res => res.json())
            .then(data => setSettings(data))
            .catch(() => { })
        fetch(`${API_URL}/api/model`)
            .then(res => res.json())
            .then(data => {
                if (data.path) {
                    const parts = data.path.replace(/\\/g, '/').split('/')
                    setModelFile(parts[parts.length - 1] || '')
                    setModelBaseDir(parts.slice(0, -1).join('/'))
                    setModelLoaded(true)
                }
            })
            .catch(() => { })
    }, [])

    useEffect(() => {
        let ws: WebSocket | null = null
        let reconnectAttempts = 0
        const maxReconnectAttempts = 10
        let reconnectTimeout: ReturnType<typeof setTimeout> | null = null
        let isMounted = true

        const connect = () => {
            if (!isMounted) return

            ws = new WebSocket(WS_URL)
            wsRef.current = ws

            ws.onopen = () => {
                setConnected(true)
                reconnectAttempts = 0
                addLog('Connected to backend')
            }

            ws.onclose = () => {
                setConnected(false)
                wsRef.current = null

                if (isMounted && reconnectAttempts < maxReconnectAttempts) {
                    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000)
                    reconnectAttempts++
                    addLog(`Disconnected. Reconnecting in ${delay / 1000}s... (${reconnectAttempts}/${maxReconnectAttempts})`)
                    reconnectTimeout = setTimeout(connect, delay)
                } else if (reconnectAttempts >= maxReconnectAttempts) {
                    addLog('Failed to connect after multiple attempts. Please restart the app.')
                }
            }

            ws.onerror = () => {
                // Error will trigger onclose, which handles reconnection
            }

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data)
                if (data.type === 'frame' && imgRef.current) {
                    imgRef.current.src = `data:image/jpeg;base64,${data.data}`
                    setStats(data.stats)

                    // Log detection events when state changes
                    const newWarning = data.warning_active || false
                    const newAlert = data.alert_active || false

                    // Minimum display time for visual indicators (5 seconds)
                    const MIN_DISPLAY_MS = 5000

                    setWarningActive(prev => {
                        if (newWarning && !prev) {
                            addLog(` WARNING: Possible drone detected (${data.stats?.warnings || 0} events)`)
                        }
                        if (newWarning) {
                            // Clear any existing timer and set new one
                            if ((window as any).__warningTimer) clearTimeout((window as any).__warningTimer)
                                ; (window as any).__warningTimer = setTimeout(() => setWarningActive(false), MIN_DISPLAY_MS)
                            return true
                        }
                        // Only turn off if no timer is running
                        return prev
                    })

                    setAlertActive(prev => {
                        if (newAlert && !prev) {
                            addLog(`  ALERT: Drone confirmed! (${data.stats?.alerts || 0} events)`)
                        }
                        if (newAlert) {
                            // Clear any existing timer and set new one
                            if ((window as any).__alertTimer) clearTimeout((window as any).__alertTimer)
                                ; (window as any).__alertTimer = setTimeout(() => setAlertActive(false), MIN_DISPLAY_MS)
                            return true
                        }
                        // Only turn off if no timer is running
                        return prev
                    })
                }
            }
        }

        // Initial connection attempt after short delay to let backend start
        setTimeout(connect, 500)

        return () => {
            isMounted = false
            if (reconnectTimeout) clearTimeout(reconnectTimeout)
            if (ws) ws.close()
        }
    }, [])

    // Theme State
    const [theme, setTheme] = useState<'dark' | 'light'>(() => {
        if (typeof window !== 'undefined') {
            return localStorage.getItem('theme') as 'dark' | 'light' || 'dark'
        }
        return 'dark'
    })

    // Apply Theme
    useEffect(() => {
        const root = window.document.documentElement
        root.classList.remove('light', 'dark')
        root.classList.add(theme)
        localStorage.setItem('theme', theme)
    }, [theme])

    const toggleTheme = () => setTheme(prev => prev === 'dark' ? 'light' : 'dark')

    // Fusion stats polling (MJPEG stream handles video natively via <img> tag)
    useEffect(() => {
        if (appMode === 'single' || !fusionStreaming) {
            setFusionConnected(false)
            return
        }
        setFusionConnected(true)

        const MIN_DISPLAY_MS = 5000
        const interval = setInterval(async () => {
            try {
                const res = await fetch(`${API_URL}/api/fusion/status`)
                if (res.ok) {
                    const data = await res.json()
                    if (data.stats && data.stats.frame_id) {
                        setFusionStats(data.stats)

                        // Wire warning/alert from fusion stats
                        const newWarning = data.stats.warning_active || false
                        const newAlert = data.stats.alert_active || false

                        setWarningActive(prev => {
                            if (newWarning && !prev) {
                                addLog(`WARNING: Possible drone detected (${data.stats.warning_events || 0} events)`)
                            }
                            if (newWarning) {
                                if ((window as any).__warningTimer) clearTimeout((window as any).__warningTimer)
                                    ; (window as any).__warningTimer = setTimeout(() => setWarningActive(false), MIN_DISPLAY_MS)
                                return true
                            }
                            return prev
                        })

                        setAlertActive(prev => {
                            if (newAlert && !prev) {
                                addLog(`ALERT: Drone confirmed! (${data.stats.alert_events || 0} events)`)
                            }
                            if (newAlert) {
                                if ((window as any).__alertTimer) clearTimeout((window as any).__alertTimer)
                                    ; (window as any).__alertTimer = setTimeout(() => setAlertActive(false), MIN_DISPLAY_MS)
                                return true
                            }
                            return prev
                        })

                        // Confuser gate suppression logging
                        const confSuppressed = data.stats.confuser_suppressed || false
                        if (confSuppressed && !(window as any).__lastConfuserSuppressed) {
                            const labels = data.stats.confuser_labels || {}
                            const parts: string[] = []
                            if (labels.rgb?.length) parts.push(`RGB: ${labels.rgb[0]}`)
                            if (labels.ir?.length) parts.push(`IR: ${labels.ir[0]}`)
                            const detail = parts.length ? ` (${parts.join(', ')})` : ''
                            addLog(`CONFUSER GATE: Alert suppressed${detail}`)
                        }
                        (window as any).__lastConfuserSuppressed = confSuppressed
                    }
                    // Detect stream end
                    if (data.streaming === false) {
                        setFusionStreaming(false)
                    }
                    setFusionConnected(true)
                }
            } catch {
                setFusionConnected(false)
            }
        }, 500)

        return () => clearInterval(interval)
    }, [appMode, fusionStreaming])

    // Fusion control functions
    const fusionSetSource = async () => {
        try {
            if (appMode === 'paired') {
                if (!fusionRgbPath || !fusionIrPath) { addLog('Set both RGB and IR video paths'); return }
                const res = await fetch(`${API_URL}/api/fusion/source/paired`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ rgb_path: fusionRgbPath, ir_path: fusionIrPath })
                })
                if (!res.ok) { addLog(`Failed: ${await res.text()}`); return }
                addLog('Paired source set')
            } else {
                if (!fusionRgbPath) { addLog('Set video path'); return }
                const res = await fetch(`${API_URL}/api/fusion/source/grayscale`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: fusionRgbPath })
                })
                if (!res.ok) { addLog(`Failed: ${await res.text()}`); return }
                addLog('Grayscale source set')
            }
            fusionStart()
        } catch { addLog('Failed to set fusion source') }
    }

    const fusionYoutube = async () => {
        if (!fusionYoutubeUrl) return
        addLog(`Downloading YouTube video (${appMode} mode)...`)
        try {
            const res = await fetch(`${API_URL}/api/fusion/source/youtube`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: fusionYoutubeUrl, mode: appMode })
            })
            if (!res.ok) { addLog(`YouTube failed: ${await res.text()}`); return }
            addLog(`YouTube downloaded, starting ${appMode} fusion`)
            fusionStart()
        } catch { addLog('YouTube download failed') }
    }

    const fusionStart = async () => {
        await fetch(`${API_URL}/api/fusion/control/start`, { method: 'POST' })
        setFusionSessionId(prev => prev + 1)  // force new MJPEG connection
        setFusionStreaming(true)
        setFusionPaused(false)
        addLog('Fusion started')
    }

    const fusionStop = async () => {
        await fetch(`${API_URL}/api/fusion/control/stop`, { method: 'POST' })
        setFusionStreaming(false)
        setFusionPaused(false)
        addLog('Fusion stopped')
    }

    const fusionTogglePause = async () => {
        const res = await fetch(`${API_URL}/api/fusion/control/pause`, { method: 'POST' })
        const data = await res.json()
        setFusionPaused(data.paused)
        addLog(data.paused ? 'Fusion paused' : 'Fusion resumed')
    }

    // Auto-connect to ESP32
    useEffect(() => {
        const autoConnect = async () => {
            try {
                await fetch(`${API_URL}/api/esp32/connect`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ address: 'http://drone-alert.local:5000' })
                })
            } catch { }
        }
        const timer = setTimeout(autoConnect, 2000)
        return () => clearTimeout(timer)
    }, [])

    const addLog = (msg: string) => {
        const time = new Date().toLocaleTimeString()
        setLogs(prev => [`[${time}] ${msg}`, ...prev.slice(0, 99)])
    }

    const exportLog = () => {
        const content = logs.join('\n')
        const blob = new Blob([content], { type: 'text/plain' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `drone_detection_log_${new Date().toISOString().slice(0, 10)}.txt`
        a.click()
        URL.revokeObjectURL(url)
    }

    const clearLogs = () => {
        setLogs([])
        localStorage.removeItem('droneDetectionLogs')
    }

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (file) setSelectedFile(file.name)
    }

    const handleModelSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (file) setModelFile(file.name)
    }

    const loadModel = async () => {
        if (!fullModelPath) { addLog('Please enter a model path'); return }
        addLog(`Loading model: ${fullModelPath}...`)
        try {
            const res = await fetch(`${API_URL}/api/model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: fullModelPath })
            })
            if (res.ok) {
                setModelLoaded(true)
                addLog(`Model loaded: ${modelFile}`)
            } else {
                const err = await res.text()
                addLog(`Failed to load model: ${err}`)
                setModelLoaded(false)
            }
        } catch { addLog('Failed to load model'); setModelLoaded(false) }
    }

    const openFile = async (path: string) => {
        if (!path) { addLog('Please enter a file path'); return }
        try {
            const res = await fetch(`${API_URL}/api/source/file`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path })
            })
            if (res.ok) {
                setSource(path.split(/[\\/]/).pop() || 'File')
                addLog('File opened')
                startStream()
            } else {
                addLog('Failed to open file')
            }
        } catch { addLog('Failed to open file') }
    }

    const openWebcam = async () => {
        try {
            const res = await fetch(`${API_URL}/api/source/webcam`, { method: 'POST' })
            if (res.ok) {
                setSource('Webcam')
                addLog('Webcam opened')
                startStream()
            }
        } catch { addLog('Failed to open webcam') }
    }

    const openYoutube = async () => {
        if (!youtubeUrl) return
        addLog('Downloading...')
        try {
            const res = await fetch(`${API_URL}/api/source/youtube`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: youtubeUrl })
            })
            if (res.ok) {
                setSource('YouTube')
                addLog('Video loaded')
                startStream()
            }
        } catch { addLog('Download failed') }
    }

    const startStream = async () => {
        await fetch(`${API_URL}/api/control/start`, { method: 'POST' })
        setStreaming(true)
        setPaused(false)
        addLog('Started')
    }

    const stopStream = async () => {
        await fetch(`${API_URL}/api/control/stop`, { method: 'POST' })
        setStreaming(false)
        setPaused(false)
        addLog('Stopped')
    }

    const togglePause = async () => {
        const res = await fetch(`${API_URL}/api/control/pause`, { method: 'POST' })
        const data = await res.json()
        setPaused(data.paused)
        addLog(data.paused ? 'Paused' : 'Resumed')
    }

    const seekVideo = async (frames: number) => {
        try {
            await fetch(`${API_URL}/api/control/seek`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frames })
            })
        } catch { }
    }

    const dismissAlert = async () => {
        await fetch(`${API_URL}/api/alert/dismiss`, { method: 'POST' })
        // Backend dismiss_alert is GLOBAL SUPPRESSION - clears both warning + alert
        setAlertActive(false)
        setWarningActive(false)
        addLog('Alert dismissed')
    }

    const openSettings = async () => {
        if (appMode === 'single') {
            setPendingSettings({ ...settings })
        } else {
            // Fusion mode: fetch live fusion config and map to Settings keys
            try {
                const res = await fetch(`${API_URL}/api/fusion/config`)
                const fc = await res.json()
                setPendingSettings({
                    ...settings,
                    infer_fps: fc.infer_fps ?? 5,
                    warning_cooldown: fc.warning_cooldown_s ?? 3.0,
                    alert_cooldown: fc.alert_cooldown_s ?? 3.0,
                    warning_window_size: fc.warning_window_frames ?? 10,
                    warning_require_hits: fc.warning_require_hits ?? 9,
                    alert_window_size: fc.alert_window_frames ?? 10,
                    alert_require_hits: fc.alert_require_hits ?? 9,
                    alert_avg_conf_threshold: fc.alert_avg_conf_threshold ?? 0.0,
                    show_gate: fc.show_gate ?? true,
                    show_troi: fc.show_troi ?? true,
                    show_cascade: fc.show_source_tags ?? true,
                    save_video: fc.save_video ?? false,
                    rgb_conf: fc.rgb_conf ?? 0.25,
                    ir_conf: fc.ir_conf ?? 0.40,
                    ir_conf_grayscale: fc.ir_conf_grayscale ?? 0.05,
                    simple_mode: fc.simple_mode ?? false,
                    use_patch_verifier: fc.use_patch_verifier ?? true,
                    patch_threshold: fc.patch_threshold ?? 0.70,
                })
            } catch {
                setPendingSettings({ ...settings })
            }
        }
        setShowSettings(true)
    }

    const updatePending = (updates: Partial<Settings>) => {
        if (pendingSettings) {
            setPendingSettings({ ...pendingSettings, ...updates })
        }
    }

    const saveSettings = async () => {
        if (!pendingSettings) return

        // Optimistic update
        setSettings(pendingSettings)
        setShowSettings(false)

        try {
            if (appMode === 'single') {
                // Single mode: POST to detection.py settings
                await fetch(`${API_URL}/api/settings`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(pendingSettings)
                })
            } else {
                // Fusion mode: POST to fusion config endpoint with mapped keys
                const fusionPayload: Record<string, unknown> = {
                    infer_fps: pendingSettings.infer_fps,
                    warning_window_frames: pendingSettings.warning_window_size,
                    warning_require_hits: pendingSettings.warning_require_hits,
                    alert_window_frames: pendingSettings.alert_window_size,
                    alert_require_hits: pendingSettings.alert_require_hits,
                    alert_avg_conf_threshold: pendingSettings.alert_avg_conf_threshold,
                    warning_cooldown_s: pendingSettings.warning_cooldown,
                    alert_cooldown_s: pendingSettings.alert_cooldown,
                    show_troi: pendingSettings.show_troi,
                    show_gate: pendingSettings.show_gate,
                    show_source_tags: pendingSettings.show_cascade, // reuse show_cascade checkbox
                    save_video: pendingSettings.save_video,
                    rgb_conf: pendingSettings.rgb_conf,
                    ir_conf: pendingSettings.ir_conf,
                    ir_conf_grayscale: pendingSettings.ir_conf_grayscale,
                    simple_mode: pendingSettings.simple_mode,
                    use_patch_verifier: pendingSettings.use_patch_verifier,
                    patch_threshold: pendingSettings.patch_threshold,
                }
                await fetch(`${API_URL}/api/fusion/config`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(fusionPayload)
                })
            }
            addLog('Settings saved')
        } catch {
            addLog('Failed to save settings')
        }
    }

    const connectEsp32 = async () => {
        try {
            const res = await fetch(`${API_URL}/api/esp32/connect`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ address: esp32Addr })
            })
            addLog(res.ok ? 'ESP32 connected' : 'ESP32 failed')
        } catch { addLog('ESP32 failed') }
    }

    return (
        <div className="min-h-screen bg-slate-100 text-slate-900 dark:bg-gradient-to-br dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 dark:text-white transition-colors duration-300">
            {/* Header */}
            <header className="bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm border-b border-slate-300 dark:border-slate-700/50 px-6 py-2 transition-colors duration-300">
                <div className="flex justify-between items-center max-w-7xl mx-auto">
                    <div className="flex items-center gap-3">
                        <div>
                            <img src={theme === 'dark' ? "/logo.png" : "/logo-black.png"} alt="TALOS" className="h-14 w-auto object-contain" />
                            <p className="text-xs text-slate-500 dark:text-slate-400 tracking-widest uppercase ml-2">Drone Detection System</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-4">
                        {alertActive && (
                            <div className="flex items-center gap-2 px-3 py-1.5 bg-red-500/20 border border-red-500/50 rounded-full animate-pulse">
                                <span className="w-2 h-2 bg-red-500 rounded-full"></span>
                                <span className="text-sm font-medium text-red-400">ALERT</span>
                            </div>
                        )}
                        {warningActive && !alertActive && (
                            <div className="flex items-center gap-2 px-3 py-1.5 bg-amber-500/20 border border-amber-500/50 rounded-full animate-pulse">
                                <span className="w-2 h-2 bg-amber-500 rounded-full"></span>
                                <span className="text-sm font-medium text-amber-400">WARNING</span>
                            </div>
                        )}
                    </div>

                    <div className="flex items-center gap-3">
                        <button onClick={toggleTheme} className="p-2 bg-slate-200 dark:bg-slate-700 border border-slate-400 dark:border-slate-600 hover:bg-slate-300 dark:hover:bg-slate-600 rounded-lg text-slate-600 dark:text-slate-200 transition-colors">
                            <ThemeIcon theme={theme} />
                        </button>
                        <button onClick={openSettings} className="p-2 bg-slate-200 dark:bg-slate-700 border border-slate-400 dark:border-slate-600 hover:bg-slate-300 dark:hover:bg-slate-600 rounded-lg text-slate-600 dark:text-slate-200 transition-colors">
                            <SettingsIcon />
                        </button>
                        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${connected ? 'bg-emerald-500/20 border border-emerald-500/50' : 'bg-red-500/20 border border-red-500/50'}`}>
                            <span className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-500' : 'bg-red-500'}`}></span>
                            <span className={`text-sm ${connected ? 'text-emerald-400' : 'text-red-400'}`}>
                                {connected ? 'Connected' : 'Disconnected'}
                            </span>
                        </div>
                    </div>
                </div>
            </header >

            <div className="max-w-7xl mx-auto p-2 lg:p-3 space-y-2">
                {/* Mode Selector */}
                <div className="flex items-center gap-1 bg-white dark:bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-400 dark:border-slate-700/50 p-1 shadow-sm dark:shadow-none transition-colors">
                    {([['single', 'Single Model'], ['paired', 'Paired Fusion'], ['grayscale', 'Grayscale Fusion']] as [AppMode, string][]).map(([mode, label]) => (
                        <button key={mode} onClick={() => setAppMode(mode)}
                            className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-colors ${appMode === mode
                                ? 'bg-cyan-500 text-white shadow-lg shadow-cyan-500/20'
                                : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700'}`}>
                            {label}
                        </button>
                    ))}
                </div>

                {/* ===== FUSION MODE VIEW ===== */}
                {appMode !== 'single' && (
                    <>
                        <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-3">
                            {/* Fusion Video */}
                            <div className="bg-white dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-400 dark:border-slate-700/50 overflow-hidden transition-colors shadow-sm dark:shadow-none">
                                <div className="relative aspect-video bg-slate-100 dark:bg-slate-900 flex items-center justify-center">
                                    {fusionStreaming ? (
                                        <img key={fusionSessionId} src={`${API_URL}/api/fusion/mjpeg?s=${fusionSessionId}`} className="w-full h-full object-contain" alt="Fusion stream" />
                                    ) : (
                                        <div className="absolute inset-0 flex items-center justify-center bg-slate-100/80 dark:bg-slate-900/80">
                                            <p className="text-slate-500 dark:text-slate-400">
                                                {appMode === 'paired' ? 'Load RGB + IR videos' : 'Load a video for grayscale fusion'}
                                            </p>
                                        </div>
                                    )}
                                </div>

                                {/* Fusion seek bar */}
                                {fusionStreaming && fusionStats && fusionStats.total_frames > 0 && (
                                    <div className="bg-white dark:bg-slate-800/80 border-t border-slate-200 dark:border-slate-700/30 px-4 py-2">
                                        <div className="flex items-center gap-3">
                                            <span className="text-xs font-mono text-slate-400 w-16 text-right">
                                                {fusionStats.video_fps > 0 ? `${Math.floor(fusionStats.frame_id / fusionStats.video_fps / 60)}:${String(Math.floor(fusionStats.frame_id / fusionStats.video_fps % 60)).padStart(2, '0')}` : '--:--'}
                                            </span>
                                            <div className="flex-1 h-1.5 rounded-full bg-slate-200 dark:bg-slate-600 overflow-hidden">
                                                <div className="h-full bg-cyan-500 transition-all duration-300"
                                                    style={{ width: `${(fusionStats.frame_id / fusionStats.total_frames) * 100}%` }} />
                                            </div>
                                            <span className="text-xs font-mono text-slate-400 w-16">
                                                {fusionStats.video_fps > 0 ? `${Math.floor(fusionStats.total_frames / fusionStats.video_fps / 60)}:${String(Math.floor(fusionStats.total_frames / fusionStats.video_fps % 60)).padStart(2, '0')}` : '--:--'}
                                            </span>
                                        </div>
                                    </div>
                                )}

                                {/* Fusion controls bar */}
                                <div className="bg-white dark:bg-slate-800/80 border-t border-slate-300 dark:border-slate-700/50 px-2 py-1.5 flex items-center justify-between transition-colors">
                                    <span className="text-sm text-slate-500 dark:text-slate-400">
                                        Mode: <span className="text-slate-900 dark:text-white capitalize">{appMode}</span>
                                    </span>
                                    <div className="flex items-center gap-1">
                                        <button onClick={fusionStreaming ? fusionTogglePause : fusionStart}
                                            className={`px-4 py-2 rounded-lg font-medium flex items-center gap-2 text-white ${fusionStreaming ? (fusionPaused ? 'bg-emerald-500' : 'bg-amber-500') : 'bg-emerald-500'}`}>
                                            {fusionStreaming ? (fusionPaused ? <><PlayIcon /> Resume</> : <><PauseIcon /> Pause</>) : <><PlayIcon /> Play</>}
                                        </button>
                                        <button onClick={fusionStop} className="bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 p-2 rounded-lg text-slate-600 dark:text-white transition-colors"><StopIcon /></button>
                                        <div className="flex items-center gap-0.5 ml-2">
                                            {[{label: '1x', val: 1}, {label: '2x', val: 2}, {label: '4x', val: 4}, {label: 'Max', val: 0}].map(({label, val}) => (
                                                <button key={label}
                                                    onClick={() => {
                                                        setFusionSpeed(val)
                                                        fetch(`${API_URL}/api/fusion/control/speed`, {
                                                            method: 'POST', headers: { 'Content-Type': 'application/json' },
                                                            body: JSON.stringify({ speed: val })
                                                        })
                                                    }}
                                                    className={`px-2 py-1 rounded text-xs font-bold transition-colors ${fusionSpeed === val
                                                        ? 'bg-cyan-500 text-white' : 'bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-600'}`}>
                                                    {label}
                                                </button>
                                            ))}
                                            <button
                                                onClick={() => fetch(`${API_URL}/api/fusion/control/skip`, {
                                                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                                                    body: JSON.stringify({ seconds: 30 })
                                                })}
                                                className="px-2 py-1 rounded text-xs font-bold bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors ml-1">
                                                Skip 30s ▶▶
                                            </button>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-4 text-sm">
                                        {fusionStats && (
                                            <>
                                                <span className="text-slate-400">Infer: <span className="text-cyan-400 font-mono">{fusionStats.infer_ms}ms</span></span>
                                                <span className="text-slate-400">Frame: <span className="text-cyan-400 font-mono">{fusionStats.frame_id}{fusionStats.total_frames > 0 ? `/${fusionStats.total_frames}` : ''}</span></span>
                                            </>
                                        )}
                                    </div>
                                </div>
                            </div>

                            {/* Right panel: Trust + Source */}
                            <div className="flex flex-col gap-3">
                                {/* Trust Decision */}
                                <div className={`backdrop-blur-sm rounded-2xl border p-4 transition-colors ${alertActive ? 'bg-red-500/20 border-red-500/50 animate-pulse' : warningActive ? 'bg-amber-500/20 border-amber-500/50' : 'bg-white dark:bg-slate-800/50 border-slate-400 dark:border-slate-700/50'}`}>
                                    <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 mb-2">Fusion Decision</h3>
                                    {fusionStats ? (
                                        <div className="space-y-3">
                                            <div className="text-center">
                                                <div className={`text-2xl font-bold uppercase ${TRUST_COLORS[fusionStats.trust_label] || 'text-slate-400'}`}>
                                                    {fusionStats.trust_name.replace(/_/g, ' ')}
                                                </div>
                                                <div className="text-xs text-slate-400 mt-1">
                                                    {(Math.max(...fusionStats.trust_probs) * 100).toFixed(1)}% confidence
                                                </div>
                                            </div>
                                            {/* Patch verifier (confuser filter) status */}
                                            {fusionStats.verifier?.active && (() => {
                                                const v = fusionStats.verifier!
                                                {/* OLD per-frame veto styling — kept for reference
                                                const vetoed = v.vetoed
                                                const borderCls = vetoed
                                                    ? 'bg-red-500/15 border-red-500/40'
                                                    : 'bg-emerald-500/10 border-emerald-500/30'
                                                const badgeCls = vetoed
                                                    ? 'bg-red-500/30 text-red-300 border-red-500/50'
                                                    : 'bg-emerald-500/30 text-emerald-300 border-emerald-500/50'
                                                */}
                                                const borderCls = 'bg-slate-500/10 border-slate-500/30'
                                                const fmtP = (p: number | null) => p == null ? '—' : p.toFixed(2)
                                                const fmtReason = (r: string | null) => {
                                                    if (!r) return null
                                                    return r === 'classifier_untrusted' ? 'classifier untrusted'
                                                        : r === 'grayscale_input' ? 'grayscale input'
                                                        : r === 'grayscale_ood' ? 'grayscale OOD'
                                                        : r === 'no_detections' ? 'no detections'
                                                        : r === 'verifier_off' ? 'verifier off'
                                                        : r
                                                }
                                                const rgbReason = fmtReason(v.rgb_skipped_reason)
                                                const irReason = fmtReason(v.ir_skipped_reason)
                                                const thr = v.threshold ?? 0.5
                                                const rgbHot = v.rgb_max_p != null && v.rgb_max_p >= thr
                                                const irHot = v.ir_max_p != null && v.ir_max_p >= thr
                                                return (
                                                    <div className={`rounded-xl border p-2.5 ${borderCls}`}>
                                                        <div className="flex items-center justify-between mb-1.5">
                                                            <span className="text-[11px] font-semibold text-slate-300 uppercase tracking-wide">Confuser Filter</span>
                                                            <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded border bg-slate-500/20 text-slate-400 border-slate-500/40">
                                                                alert gate
                                                            </span>
                                                        </div>
                                                        {(rgbHot || irHot) && (
                                                            <div className="text-[10px] text-amber-400 mb-1.5">
                                                                ⚠ confuser detected — will suppress alert if persistent
                                                            </div>
                                                        )}
                                                        {/* OLD per-frame trust downgrade display — kept for reference
                                                        {vetoed && (
                                                            <div className="text-[10px] text-slate-400 mb-1.5">
                                                                was <span className="font-mono text-slate-300">{v.original_trust_name.replace(/_/g, ' ')}</span> → <span className="font-mono text-slate-300">{fusionStats.trust_name.replace(/_/g, ' ')}</span>
                                                            </div>
                                                        )}
                                                        */}
                                                        <div className="space-y-0.5 text-[10px]">
                                                            <div className="flex justify-between">
                                                                <span className="text-slate-500">RGB</span>
                                                                {rgbReason ? (
                                                                    <span className="text-slate-500 italic">{rgbReason}</span>
                                                                ) : (
                                                                    <span className={`font-mono ${v.rgb_max_p != null && v.rgb_max_p >= thr ? 'text-red-300' : 'text-emerald-300'}`}>
                                                                        P(conf)={fmtP(v.rgb_max_p)} · {v.rgb_n_boxes} box{v.rgb_n_boxes === 1 ? '' : 'es'}
                                                                    </span>
                                                                )}
                                                            </div>
                                                            {v.rgb_labels && v.rgb_labels.length > 0 && (
                                                                <div className="text-[9px] font-mono text-slate-400 pl-8 -mt-0.5">{v.rgb_labels.slice(0, 3).join(', ')}</div>
                                                            )}
                                                            <div className="flex justify-between">
                                                                <span className="text-slate-500">IR</span>
                                                                {irReason ? (
                                                                    <span className="text-slate-500 italic">{irReason}</span>
                                                                ) : (
                                                                    <span className={`font-mono ${v.ir_max_p != null && v.ir_max_p >= thr ? 'text-red-300' : 'text-emerald-300'}`}>
                                                                        P(conf)={fmtP(v.ir_max_p)} · {v.ir_n_boxes} box{v.ir_n_boxes === 1 ? '' : 'es'}
                                                                    </span>
                                                                )}
                                                            </div>
                                                            {v.ir_labels && v.ir_labels.length > 0 && (
                                                                <div className="text-[9px] font-mono text-slate-400 pl-8 -mt-0.5">{v.ir_labels.slice(0, 3).join(', ')}</div>
                                                            )}
                                                        </div>
                                                    </div>
                                                )
                                            })()}
                                            {/* Probability bars */}
                                            <div className="space-y-1">
                                                {['reject_both', 'trust_rgb', 'trust_ir', 'trust_both'].map((name, i) => (
                                                    <div key={name} className="flex items-center gap-2">
                                                        <span className="text-[10px] text-slate-500 w-16 text-right truncate">{name.replace('_', ' ')}</span>
                                                        <div className="flex-1 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                                                            <div className={`h-full rounded-full transition-all duration-300 ${i === 0 ? 'bg-red-500' : i === 1 ? 'bg-blue-500' : i === 2 ? 'bg-orange-500' : 'bg-emerald-500'}`}
                                                                style={{ width: `${(fusionStats.trust_probs[i] || 0) * 100}%` }} />
                                                        </div>
                                                        <span className="text-[10px] font-mono text-slate-400 w-10">{((fusionStats.trust_probs[i] || 0) * 100).toFixed(1)}%</span>
                                                    </div>
                                                ))}
                                            </div>
                                            {/* Detection counts */}
                                            <div className="grid grid-cols-2 gap-2 text-center">
                                                <div className="bg-slate-100/50 dark:bg-slate-900/30 rounded-lg p-2">
                                                    <div className="text-lg font-bold text-blue-400">{fusionStats.rgb_dets}</div>
                                                    <div className="text-[10px] text-slate-500">RGB Dets</div>
                                                </div>
                                                <div className="bg-slate-100/50 dark:bg-slate-900/30 rounded-lg p-2">
                                                    <div className="text-lg font-bold text-orange-400">{fusionStats.ir_dets}</div>
                                                    <div className="text-[10px] text-slate-500">IR Dets</div>
                                                </div>
                                            </div>
                                            {/* Warning/Alert event counts */}
                                            <div className="grid grid-cols-2 gap-2 text-center">
                                                <div className={`rounded-lg p-2 ${fusionStats.warning_active ? 'bg-amber-500/20 border border-amber-500/40' : 'bg-slate-100/50 dark:bg-slate-900/30'}`}>
                                                    <div className={`text-lg font-bold ${fusionStats.warning_active ? 'text-amber-400' : 'text-slate-400'}`}>{fusionStats.warning_events || 0}</div>
                                                    <div className="text-[10px] text-slate-500">Warnings</div>
                                                </div>
                                                <div className={`rounded-lg p-2 ${fusionStats.alert_active ? 'bg-red-500/20 border border-red-500/40 animate-pulse' : 'bg-slate-100/50 dark:bg-slate-900/30'}`}>
                                                    <div className={`text-lg font-bold ${fusionStats.alert_active ? 'text-red-400' : 'text-slate-400'}`}>{fusionStats.alert_events || 0}</div>
                                                    <div className="text-[10px] text-slate-500">Alerts</div>
                                                </div>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="text-center text-slate-500 py-4">No fusion data</div>
                                    )}
                                </div>

                                {/* Fusion Source */}
                                <div className="bg-white dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-400 dark:border-slate-700/50 p-4 flex-1 transition-colors shadow-sm dark:shadow-none">
                                    <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 mb-3">
                                        {appMode === 'paired' ? 'Paired Video Source' : 'Video Source'}
                                    </h3>
                                    <div className="space-y-2">
                                        {/* Base directory */}
                                        <input value={fusionBaseDir} onChange={e => setFusionBaseDir(e.target.value)}
                                            placeholder="Base directory (e.g. C:\Videos)"
                                            className="w-full bg-slate-50 dark:bg-slate-900/50 border border-slate-400 dark:border-slate-700 rounded-xl px-2 py-1.5 text-xs text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 transition-colors" />
                                        {/* RGB / main video browse */}
                                        <div className="flex gap-2">
                                            <input ref={fusionRgbInputRef} type="file" accept="video/*" onChange={e => {
                                                const f = e.target.files?.[0]
                                                if (f) {
                                                    const dir = fusionBaseDir.replace(/\\/g, '/')
                                                    const sep = dir.endsWith('/') ? '' : '/'
                                                    setFusionRgbPath(`${dir}${sep}${f.name}`)
                                                }
                                            }} className="hidden" />
                                            <input value={fusionRgbPath} onChange={e => setFusionRgbPath(e.target.value)}
                                                placeholder={appMode === 'paired' ? 'RGB video path' : 'Video path'}
                                                className="flex-1 bg-slate-50 dark:bg-slate-900/50 border border-slate-400 dark:border-slate-700 rounded-xl px-2 py-1.5 text-xs text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 transition-colors" />
                                            <button onClick={() => fusionRgbInputRef.current?.click()}
                                                className="bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 border border-slate-400 dark:border-slate-600 px-3 rounded-xl text-xs font-medium text-slate-700 dark:text-white transition-colors">
                                                <FolderIcon />
                                            </button>
                                        </div>
                                        {/* IR video browse (paired mode only) */}
                                        {appMode === 'paired' && (
                                            <div className="flex gap-2">
                                                <input ref={fusionIrInputRef} type="file" accept="video/*" onChange={e => {
                                                    const f = e.target.files?.[0]
                                                    if (f) {
                                                        const dir = fusionBaseDir.replace(/\\/g, '/')
                                                        const sep = dir.endsWith('/') ? '' : '/'
                                                        setFusionIrPath(`${dir}${sep}${f.name}`)
                                                    }
                                                }} className="hidden" />
                                                <input value={fusionIrPath} onChange={e => setFusionIrPath(e.target.value)}
                                                    placeholder="IR video path"
                                                    className="flex-1 bg-slate-50 dark:bg-slate-900/50 border border-slate-400 dark:border-slate-700 rounded-xl px-2 py-1.5 text-xs text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 transition-colors" />
                                                <button onClick={() => fusionIrInputRef.current?.click()}
                                                    className="bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 border border-slate-400 dark:border-slate-600 px-3 rounded-xl text-xs font-medium text-slate-700 dark:text-white transition-colors">
                                                    <FolderIcon />
                                                </button>
                                            </div>
                                        )}
                                        <button onClick={fusionSetSource}
                                            className="w-full bg-cyan-500 hover:bg-cyan-600 text-white py-1.5 rounded-xl text-sm font-medium transition-colors">
                                            Load & Start
                                        </button>
                                        {/* YouTube — available for grayscale and single fusion modes */}
                                        <div className="border-t border-slate-200 dark:border-slate-700 pt-2 mt-2">
                                            <span className="text-[10px] text-slate-500 uppercase tracking-wider">or YouTube</span>
                                        </div>
                                        <div className="flex gap-2">
                                            <input value={fusionYoutubeUrl} onChange={e => setFusionYoutubeUrl(e.target.value)}
                                                placeholder="YouTube URL"
                                                className="flex-1 bg-slate-50 dark:bg-slate-900/50 border border-slate-400 dark:border-slate-700 rounded-xl px-2 py-1.5 text-xs text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 transition-colors" />
                                            <button onClick={fusionYoutube} className="bg-red-600 text-white px-3 rounded-xl shadow-lg shadow-red-900/20"><PlayIcon /></button>
                                        </div>
                                    </div>
                                </div>

                                {/* Fusion WS status */}
                                <div className={`flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-xs ${fusionConnected ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400' : 'bg-red-500/10 border border-red-500/30 text-red-400'}`}>
                                    <span className={`w-1.5 h-1.5 rounded-full ${fusionConnected ? 'bg-emerald-500' : 'bg-red-500'}`}></span>
                                    Fusion WS {fusionConnected ? 'Connected' : 'Disconnected'}
                                </div>
                            </div>
                        </div>
                    </>
                )}

                {/* ===== SINGLE MODEL VIEW (original) ===== */}
                {appMode === 'single' && (
                <>
                <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-3">
                    <div className={`${isFullscreen ? 'fixed inset-0 z-50 bg-black flex flex-col' : 'bg-white dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-400 dark:border-slate-700/50 overflow-hidden transition-colors shadow-sm dark:shadow-none'}`}>
                        <div className={`relative transition-colors ${isFullscreen ? 'flex-1 flex items-center justify-center bg-black' : 'aspect-video bg-slate-100 dark:bg-slate-900 flex items-center justify-center'}`}>

                            {/* Fullscreen Toggle */}
                            <button
                                onClick={() => setIsFullscreen(!isFullscreen)}
                                className="absolute top-4 right-4 z-10 p-2 bg-black/50 hover:bg-black/70 text-white rounded-lg backdrop-blur-sm transition-colors"
                                title={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}
                            >
                                {isFullscreen ? <MinimizeIcon /> : <MaximizeIcon />}
                            </button>

                            <img ref={imgRef} className={`${isFullscreen ? 'h-full w-full' : 'w-full h-full'} object-contain`} alt="" />
                            {!streaming && (
                                <div className="absolute inset-0 flex items-center justify-center bg-slate-100/80 dark:bg-slate-900/80 transition-colors">
                                    <p className="text-slate-500 dark:text-slate-400">Select a source and press Play</p>
                                </div>
                            )}
                        </div>
                        {/* Seek bar */}
                        {streaming && stats.total_frames > 0 && (
                            <div className={`${isFullscreen ? 'bg-slate-900/90 absolute bottom-12 w-full' : 'bg-white dark:bg-slate-800/80 border-t border-slate-200 dark:border-slate-700/30'} px-4 py-2 transition-colors`}>
                                <div className="flex items-center gap-3">
                                    <span className="text-xs font-mono text-slate-400 w-16 text-right">
                                        {stats.video_fps > 0 ? `${Math.floor(stats.frame / stats.video_fps / 60)}:${String(Math.floor(stats.frame / stats.video_fps % 60)).padStart(2, '0')}` : '--:--'}
                                    </span>
                                    <input type="range" min="0" max={stats.total_frames} value={stats.frame}
                                        onChange={e => {
                                            const frame = parseInt(e.target.value)
                                            fetch(`${API_URL}/api/control/seek`, {
                                                method: 'POST',
                                                headers: { 'Content-Type': 'application/json' },
                                                body: JSON.stringify({ frames: frame, absolute: true })
                                            })
                                        }}
                                        className="flex-1 h-1.5 rounded-full appearance-none cursor-pointer accent-cyan-500
                                            [&::-webkit-slider-runnable-track]:rounded-full [&::-webkit-slider-runnable-track]:bg-slate-200 dark:[&::-webkit-slider-runnable-track]:bg-slate-600
                                            [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-cyan-500" />
                                    <span className="text-xs font-mono text-slate-400 w-16">
                                        {stats.video_fps > 0 ? `${Math.floor(stats.total_frames / stats.video_fps / 60)}:${String(Math.floor(stats.total_frames / stats.video_fps % 60)).padStart(2, '0')}` : '--:--'}
                                    </span>
                                </div>
                            </div>
                        )}
                        <div className={`${isFullscreen ? 'bg-slate-900/90 border-t border-slate-700/50 absolute bottom-0 w-full' : 'bg-white dark:bg-slate-800/80 border-t border-slate-300 dark:border-slate-700/50'} px-2 py-1.5 flex items-center justify-between transition-colors`}>
                            <span className="text-sm text-slate-500 dark:text-slate-400">Source: <span className="text-slate-900 dark:text-white">{source || 'None'}</span></span>
                            <div className="flex items-center gap-1">
                                <button onClick={streaming ? togglePause : startStream} className={`px-4 py-2 rounded-lg font-medium flex items-center gap-2 text-white ${streaming ? (paused ? 'bg-emerald-500' : 'bg-amber-500') : 'bg-emerald-500'}`}>
                                    {streaming ? (paused ? <><PlayIcon /> Resume</> : <><PauseIcon /> Pause</>) : <><PlayIcon /> Play</>}
                                </button>
                                <button onClick={stopStream} className="bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 p-2 rounded-lg text-slate-600 dark:text-white transition-colors"><StopIcon /></button>
                                {/* Speed buttons */}
                                <div className="flex items-center gap-0.5 ml-2">
                                    {[1, 2, 4].map(speed => (
                                        <button key={speed}
                                            onClick={() => {
                                                setPlaybackSpeed(speed)
                                                fetch(`${API_URL}/api/control/speed`, {
                                                    method: 'POST',
                                                    headers: { 'Content-Type': 'application/json' },
                                                    body: JSON.stringify({ speed })
                                                })
                                            }}
                                            className={`px-2 py-1 rounded text-xs font-bold transition-colors ${playbackSpeed === speed
                                                ? 'bg-cyan-500 text-white'
                                                : 'bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-600'
                                                }`}>
                                            {speed}x
                                        </button>
                                    ))}
                                </div>
                            </div>
                            <div className="flex items-center gap-4 text-sm">
                                <select value={streamQuality}
                                    onChange={e => { setStreamQuality(e.target.value); fetch(`${API_URL}/api/stream/quality?quality=${e.target.value}`, { method: 'POST' }) }}
                                    className="bg-slate-100 dark:bg-slate-700 border border-slate-400 dark:border-slate-600 rounded-lg px-2 py-1 text-xs text-slate-700 dark:text-white transition-colors">
                                    <option value="360p">360p</option>
                                    <option value="480p">480p</option>
                                    <option value="720p">720p</option>
                                    <option value="1080p">1080p</option>
                                </select>
                                <button
                                    onClick={async () => {
                                        try {
                                            const newDevice = device === 'cpu' ? 'gpu' : 'cpu';
                                            console.log(`Switching to ${newDevice}...`);
                                            const res = await fetch(`${API_URL}/api/device?device=${newDevice}`, { method: 'POST' });
                                            console.log(`Response status: ${res.status}`);
                                            if (res.ok) {
                                                setDevice(newDevice);
                                                addLog(`Switched to ${newDevice.toUpperCase()}`);
                                            } else {
                                                const err = await res.text();
                                                console.error(`Failed: ${err}`);
                                                addLog(`Failed to switch device: ${err}`);
                                            }
                                        } catch (e) {
                                            console.error('Device switch error:', e);
                                            addLog(`Device switch error: ${e}`);
                                        }
                                    }}
                                    className={`px-2 py-1 rounded-lg text-xs font-medium transition-colors ${device === 'gpu' ? 'bg-emerald-500 dark:bg-emerald-600 text-white' : 'bg-slate-200 dark:bg-slate-600 text-slate-700 dark:text-white'}`}>
                                    {device.toUpperCase()}
                                </button>
                                <span className="text-slate-400">Infer FPS: <span className="text-cyan-400 font-mono w-8 inline-block">{stats.fps}</span></span>
                                <span className="text-slate-400">Frame: <span className="text-cyan-400 font-mono w-16 inline-block">{stats.frame}{stats.total_frames > 0 ? `/${stats.total_frames}` : ''}</span></span>
                            </div>
                        </div>
                    </div>

                    {/* Right: Alert + Video Source */}
                    <div className="flex flex-col gap-4">
                        <div className={`backdrop-blur-sm rounded-2xl border p-4 ${alertActive ? 'bg-red-900/30 border-red-500/50 animate-pulse' : warningActive ? 'bg-amber-900/30 border-amber-500/50' : 'bg-white dark:bg-slate-800/50 border-slate-400 dark:border-slate-700/50 shadow-sm dark:shadow-none'} transition-colors`}>
                            <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 mb-3">Alert Control</h3>
                            <button onClick={dismissAlert} disabled={!alertActive}
                                className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl font-semibold ${alertActive ? 'bg-gradient-to-r from-red-500 to-rose-600 text-white' : 'bg-slate-100 dark:bg-slate-700 text-slate-400 dark:text-slate-500 cursor-not-allowed'} transition-colors`}>
                                <AlertIcon /> Dismiss Alert
                            </button>
                        </div>

                        {/* Model Source */}
                        <div className="bg-white dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-400 dark:border-slate-700/50 p-4 transition-colors shadow-sm dark:shadow-none">
                            <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 mb-3">Model Weights</h3>
                            <div className="space-y-2">
                                <button onClick={() => modelInputRef.current?.click()} className="w-full flex items-center justify-center gap-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 border border-slate-400 dark:border-slate-600 py-2 rounded-xl font-medium text-slate-700 dark:text-white transition-colors">
                                    <FolderIcon /> Browse Model (.pt)
                                </button>
                                <input value={modelBaseDir} onChange={e => setModelBaseDir(e.target.value)} placeholder="Model directory"
                                    className="w-full bg-slate-50 dark:bg-slate-900/50 border border-slate-400 dark:border-slate-700 rounded-xl px-2 py-1.5 text-xs text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 transition-colors" />
                                <div className="flex gap-2">
                                    <input ref={modelInputRef} type="file" accept=".pt" onChange={handleModelSelect} className="hidden" />
                                    <input value={modelFile} readOnly placeholder="Select weights..."
                                        className="w-full bg-slate-50 dark:bg-slate-900/50 border border-slate-400 dark:border-slate-700 rounded-xl px-2 py-1.5 text-xs text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 transition-colors" />
                                </div>
                                <button onClick={loadModel} disabled={!fullModelPath}
                                    className={`w-full py-1.5 rounded-xl text-sm font-medium transition-colors ${modelLoaded ? 'bg-slate-200 dark:bg-slate-700/80 border border-slate-400 dark:border-slate-600/50 text-slate-600 dark:text-slate-300 cursor-default' : 'bg-slate-100 dark:bg-slate-700 disabled:opacity-50 hover:bg-slate-200 dark:hover:bg-slate-600 border border-slate-400 dark:border-slate-600 text-slate-700 dark:text-white'}`}>
                                    {modelLoaded ? `Loaded: ${modelFile}` : 'Load Model'}
                                </button>
                            </div>
                        </div>

                        <div className="bg-white dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-400 dark:border-slate-700/50 p-4 flex-1 transition-colors shadow-sm dark:shadow-none">
                            <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 mb-3">Video Source</h3>
                            <div className="space-y-2">
                                <button onClick={() => fileInputRef.current?.click()} className="w-full flex items-center justify-center gap-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 border border-slate-400 dark:border-slate-600 py-2 rounded-xl font-medium text-slate-700 dark:text-white transition-colors">
                                    <FolderIcon /> Browse Video
                                </button>
                                <input value={baseDir} onChange={e => setBaseDir(e.target.value)} placeholder="Base directory"
                                    className="w-full bg-slate-50 dark:bg-slate-900/50 border border-slate-400 dark:border-slate-700 rounded-xl px-2 py-1.5 text-xs text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 transition-colors" />
                                <div className="flex gap-2">
                                    <input ref={fileInputRef} type="file" accept="video/*" onChange={handleFileSelect} className="hidden" />
                                    <input value={selectedFile} readOnly placeholder="Select file..."
                                        className="w-full bg-slate-50 dark:bg-slate-900/50 border border-slate-400 dark:border-slate-700 rounded-xl px-2 py-1.5 text-xs text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 transition-colors" />
                                </div>
                                <button onClick={() => openFile(fullPath)} disabled={!fullPath}
                                    className="w-full bg-slate-100 dark:bg-slate-700 disabled:opacity-50 hover:bg-slate-200 dark:hover:bg-slate-600 border border-slate-400 dark:border-slate-600 py-1.5 rounded-xl text-sm font-medium text-slate-700 dark:text-white transition-colors">
                                    Load File
                                </button>
                                <div className="flex gap-2">
                                    <input value={youtubeUrl} onChange={e => setYoutubeUrl(e.target.value)} placeholder="YouTube URL"
                                        className="flex-1 bg-slate-50 dark:bg-slate-900/50 border border-slate-400 dark:border-slate-700 rounded-xl px-2 py-1.5 text-xs text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 transition-colors" />
                                    <button onClick={openYoutube} className="bg-red-600 text-white px-3 rounded-xl shadow-lg shadow-red-900/20"><PlayIcon /></button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Statistics - single mode only */}
                <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-4">
                    <div className="bg-white dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-400 dark:border-slate-700/50 p-4 transition-colors shadow-sm dark:shadow-none">
                        <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 mb-3">Statistics</h3>
                        {/* Window Progress Bars */}
                        <div className="grid grid-cols-2 gap-3 mb-3">
                            <div className="bg-amber-100 dark:bg-amber-900/30 border border-amber-400 dark:border-amber-500/30 rounded-lg p-2 transition-colors">
                                <div className="text-xs text-amber-600 dark:text-amber-400 mb-1">
                                    Warning: {stats.warn_hits}/{stats.warn_need} hits
                                </div>
                                <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden transition-colors">
                                    <div className="h-full bg-amber-500 transition-all duration-300"
                                        style={{ width: `${Math.min(100, (stats.warn_hits / stats.warn_need) * 100)}%` }} />
                                </div>
                            </div>
                            <div className="bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-500/30 rounded-lg p-2 transition-colors">
                                <div className="text-xs text-red-600 dark:text-red-400 mb-1">
                                    Alert: {stats.alert_hits}/{stats.alert_need} hits
                                </div>
                                <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden transition-colors">
                                    <div className="h-full bg-red-500 transition-all duration-300"
                                        style={{ width: `${Math.min(100, (stats.alert_hits / stats.alert_need) * 100)}%` }} />
                                </div>
                            </div>
                        </div>
                        <div className="grid grid-cols-4 gap-3">
                            <div className="bg-slate-50 dark:bg-slate-900/50 border border-slate-400 dark:border-slate-800 rounded-xl p-3 text-center transition-colors">
                                <div className="text-2xl font-bold text-cyan-600 dark:text-cyan-400">{stats.fps}</div>
                                <div className="text-xs text-slate-500">Infer FPS</div>
                            </div>
                            <div className="bg-slate-50 dark:bg-slate-900/50 border border-slate-400 dark:border-slate-800 rounded-xl p-3 text-center transition-colors">
                                <div className="text-2xl font-bold text-cyan-500 dark:text-cyan-400">{stats.frame}</div>
                                <div className="text-xs text-slate-500">Frames</div>
                            </div>

                            <div className="bg-slate-50 dark:bg-slate-900/50 border border-amber-400 dark:border-slate-800 rounded-xl p-3 text-center transition-colors">
                                <div className="text-2xl font-bold text-amber-500 dark:text-amber-400">{stats.warnings}</div>
                                <div className="text-xs text-slate-500">Warnings</div>
                            </div>
                            <div className="bg-slate-50 dark:bg-slate-900/50 border border-red-400 dark:border-slate-800 rounded-xl p-3 text-center transition-colors">
                                <div className="text-2xl font-bold text-red-500 dark:text-red-400">{stats.alerts}</div>
                                <div className="text-xs text-slate-500">Alerts</div>
                            </div>
                        </div>
                    </div>

                    {/* ESP32 */}
                    <div className="bg-white dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-400 dark:border-slate-700/50 p-4 transition-colors shadow-sm dark:shadow-none">
                        <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 mb-3">ESP32</h3>
                        <div className="space-y-3">
                            <input value={esp32Addr} onChange={e => setEsp32Addr(e.target.value)}
                                className="w-full bg-slate-50 dark:bg-slate-900/50 border border-slate-400 dark:border-slate-700 rounded-xl px-3 py-2 text-sm text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 transition-colors" />
                            <button onClick={connectEsp32} className="w-full flex items-center justify-center gap-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 py-2.5 rounded-xl font-medium text-slate-700 dark:text-white transition-colors">
                                <WifiIcon /> Connect
                            </button>
                        </div>
                    </div>
                </div>
                </>
                )}

                {/* Event Log */}
                <div className="bg-white dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-400 dark:border-slate-700/50 p-4 transition-colors shadow-sm dark:shadow-none">
                    <div className="flex items-center justify-between mb-3">
                        <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400">Event Log</h3>
                        <div className="flex gap-2">
                            <button onClick={clearLogs} className="text-xs bg-slate-100 dark:bg-slate-700 hover:bg-red-500 hover:text-white dark:hover:bg-red-600 px-2 py-1 rounded-lg text-slate-600 dark:text-slate-300 transition-colors">Clear</button>
                            <button onClick={exportLog} className="flex items-center gap-1 text-xs bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 px-2 py-1 rounded-lg text-slate-600 dark:text-slate-300 transition-colors">
                                <DownloadIcon /> Export
                            </button>
                        </div>
                    </div>
                    <div className="h-32 overflow-y-auto font-mono text-xs space-y-1">
                        {logs.map((log, i) => (
                            <div key={i} className={log.includes('DRONE') ? 'text-red-600 dark:text-red-400 font-semibold' : log.includes('CONFUSER') ? 'text-purple-600 dark:text-purple-400 font-semibold' : log.includes('WARNING') ? 'text-amber-600 dark:text-amber-400' : 'text-emerald-600 dark:text-emerald-400 opacity-90'}>{log}</div>
                        ))}
                        {logs.length === 0 && <span className="text-slate-400 dark:text-slate-600">No events yet</span>}
                    </div>
                </div>
            </div>

            {/* Settings Modal */}
            {
                showSettings && (
                    <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm flex items-center justify-center z-50 transition-colors">
                        <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-400 dark:border-slate-700 p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto shadow-2xl transition-colors">
                            <div className="flex justify-between items-center mb-6">
                                <div className="flex items-center gap-3">
                                    <h2 className="text-xl font-bold text-slate-900 dark:text-white">Settings</h2>
                                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                                        appMode === 'single'
                                            ? 'bg-cyan-500/20 text-cyan-600 dark:text-cyan-400'
                                            : 'bg-emerald-500/20 text-emerald-600 dark:text-emerald-400'
                                    }`}>{appMode === 'single' ? 'Single Mode' : `Fusion (${appMode})`}</span>
                                </div>
                                <button onClick={() => setShowSettings(false)} className="text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-white text-2xl transition-colors">&times;</button>
                            </div>

                            <div className="space-y-6">
                                {/* Detection Settings */}
                                <div>
                                    <h3 className="text-sm font-semibold text-cyan-600 dark:text-cyan-400 mb-3">Detection</h3>

                                    {/* Simple Mode Toggle - all modes */}
                                    <div className="mb-4 p-3 rounded-lg border-2 border-dashed border-cyan-500/30 bg-cyan-500/5">
                                        <label className="flex items-center gap-3 cursor-pointer">
                                            <input type="checkbox" checked={pendingSettings?.simple_mode}
                                                onChange={e => updatePending({ simple_mode: e.target.checked })}
                                                className="w-5 h-5 rounded border-slate-400 dark:border-slate-600 text-cyan-600 focus:ring-cyan-500" />
                                            <div>
                                                <span className="text-sm font-semibold text-slate-700 dark:text-slate-200">Simple Mode</span>
                                                <p className="text-xs text-slate-500 dark:text-slate-400">
                                                    {appMode === 'single' ? 'Basic YOLO detection only' : 'Classifier only, no temporal logic'}
                                                </p>
                                            </div>
                                        </label>
                                    </div>

                                    {/* Confuser Filter Toggle - fusion modes only */}
                                    {appMode !== 'single' && (
                                    <div className="mb-4 p-3 rounded-lg border-2 border-dashed border-amber-500/30 bg-amber-500/5">
                                        <label className="flex items-center gap-3 cursor-pointer">
                                            <input type="checkbox" checked={pendingSettings?.use_patch_verifier}
                                                onChange={e => updatePending({ use_patch_verifier: e.target.checked })}
                                                className="w-5 h-5 rounded border-slate-400 dark:border-slate-600 text-amber-600 focus:ring-amber-500" />
                                            <div>
                                                <span className="text-sm font-semibold text-slate-700 dark:text-slate-200">Confuser Filter</span>
                                                <p className="text-xs text-slate-500 dark:text-slate-400">CNN rejects airplanes, helicopters, and birds</p>
                                            </div>
                                        </label>
                                        {pendingSettings?.use_patch_verifier && (
                                        <div className="mt-3 flex items-center gap-3">
                                            <label className="text-xs text-slate-500 dark:text-slate-400 whitespace-nowrap">Veto Threshold</label>
                                            <input type="number" min="0.05" max="0.99" step="0.05"
                                                value={pendingSettings?.patch_threshold}
                                                onChange={e => updatePending({ patch_threshold: parseFloat(e.target.value) })}
                                                className="w-24 bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-2 py-1 text-sm text-slate-900 dark:text-white transition-colors" />
                                            <span className="text-[10px] text-slate-400">P(confuser) ≥ threshold → veto</span>
                                        </div>
                                        )}
                                    </div>
                                    )}

                                    <div className="grid grid-cols-2 gap-4">
                                        {appMode === 'single' && (
                                        <div>
                                            <label className="text-xs text-slate-500 dark:text-slate-400">Cascade Mode</label>
                                            <p className="text-xs text-slate-500 dark:text-slate-600 mb-1">Re-verify detections with focused crop</p>
                                            <select value={pendingSettings?.cascade_mode}
                                                onChange={e => updatePending({ cascade_mode: e.target.value })}
                                                className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-3 py-2 text-slate-900 dark:text-white transition-colors">
                                                <option value="None">None</option>
                                                <option value="Low-Small">Low-Small</option>
                                                <option value="All">All</option>
                                                <option value="Alert-Window">Alert-Window</option>
                                            </select>
                                        </div>
                                        )}
                                        <div>
                                            <label className="text-xs text-slate-500 dark:text-slate-400">Inference FPS</label>
                                            <p className="text-xs text-slate-500 dark:text-slate-600 mb-1">Frames processed per second</p>
                                            <input type="number" min="1" max="30" value={pendingSettings?.infer_fps}
                                                onChange={e => updatePending({ infer_fps: parseInt(e.target.value) })}
                                                className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-3 py-2 text-slate-900 dark:text-white transition-colors" />
                                        </div>
                                        {appMode !== 'single' && (
                                        <div>
                                            <label className="text-xs text-slate-500 dark:text-slate-400">RGB Model Confidence</label>
                                            <p className="text-xs text-slate-500 dark:text-slate-600 mb-1">RGB YOLO threshold (default 0.25)</p>
                                            <input type="number" min="0.01" max="1" step="0.05" value={pendingSettings?.rgb_conf}
                                                onChange={e => updatePending({ rgb_conf: parseFloat(e.target.value) })}
                                                className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-3 py-2 text-slate-900 dark:text-white transition-colors" />
                                        </div>
                                        )}
                                        {appMode === 'paired' && (
                                        <div>
                                            <label className="text-xs text-slate-500 dark:text-slate-400">IR Model Confidence</label>
                                            <p className="text-xs text-slate-500 dark:text-slate-600 mb-1">IR YOLO threshold — paired IR video (default 0.40)</p>
                                            <input type="number" min="0.01" max="1" step="0.05" value={pendingSettings?.ir_conf}
                                                onChange={e => updatePending({ ir_conf: parseFloat(e.target.value) })}
                                                className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-3 py-2 text-slate-900 dark:text-white transition-colors" />
                                        </div>
                                        )}
                                        {appMode === 'grayscale' && (
                                        <div>
                                            <label className="text-xs text-slate-500 dark:text-slate-400">IR Model Confidence (Grayscale)</label>
                                            <p className="text-xs text-slate-500 dark:text-slate-600 mb-1">IR YOLO on converted grayscale (default 0.05)</p>
                                            <input type="number" min="0.01" max="1" step="0.01" value={pendingSettings?.ir_conf_grayscale}
                                                onChange={e => updatePending({ ir_conf_grayscale: parseFloat(e.target.value) })}
                                                className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-3 py-2 text-slate-900 dark:text-white transition-colors" />
                                        </div>
                                        )}
                                        {appMode === 'single' && (<>
                                        <div>
                                            <label className="text-xs text-slate-500 dark:text-slate-400">Detection Confidence</label>
                                            <p className="text-xs text-slate-500 dark:text-slate-600 mb-1">Minimum YOLO confidence</p>
                                            <input type="number" min="0.1" max="1" step="0.05" value={pendingSettings?.detect_conf}
                                                onChange={e => updatePending({ detect_conf: parseFloat(e.target.value) })}
                                                className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-3 py-2 text-slate-900 dark:text-white transition-colors" />
                                        </div>
                                        <div>
                                            <label className="text-xs text-slate-500 dark:text-slate-400">ROI Size</label>
                                            <p className="text-xs text-slate-500 dark:text-slate-600 mb-1">Crop size for focused inference</p>
                                            <input type="number" min="320" max="1280" step="64" value={pendingSettings?.roi_size}
                                                onChange={e => updatePending({ roi_size: parseInt(e.target.value) })}
                                                className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-3 py-2 text-slate-900 dark:text-white transition-colors" />
                                        </div>
                                        <div>
                                            <label className="text-xs text-slate-500 dark:text-slate-400">Cascade Trigger Conf</label>
                                            <p className="text-xs text-slate-500 dark:text-slate-600 mb-1">Min conf to trigger cascade verification</p>
                                            <input type="number" min="0.1" max="1" step="0.05" value={pendingSettings?.cascade_trigger_conf}
                                                onChange={e => updatePending({ cascade_trigger_conf: parseFloat(e.target.value) })}
                                                className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-3 py-2 text-slate-900 dark:text-white transition-colors" />
                                        </div>
                                        <div>
                                            <label className="text-xs text-slate-500 dark:text-slate-400">Cascade Accept Conf</label>
                                            <p className="text-xs text-slate-500 dark:text-slate-600 mb-1">Min conf to accept cascade result</p>
                                            <input type="number" min="0.1" max="1" step="0.05" value={pendingSettings?.cascade_accept_conf}
                                                onChange={e => updatePending({ cascade_accept_conf: parseFloat(e.target.value) })}
                                                className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-3 py-2 text-slate-900 dark:text-white transition-colors" />
                                        </div>
                                        <div>
                                            <label className="text-xs text-slate-500 dark:text-slate-400">Alert Min Size (W × H px)</label>
                                            <p className="text-xs text-slate-500 dark:text-slate-600 mb-1">Min detection size to trigger alert</p>
                                            <div className="flex items-center gap-2">
                                                <input type="number" min="1" max="200" step="1"
                                                    value={Math.round(Math.sqrt(pendingSettings?.alert_min_area || 625))}
                                                    onChange={e => {
                                                        const w = parseInt(e.target.value) || 25
                                                        const hInput = (e.target as HTMLInputElement).parentElement?.querySelector('[data-dim="h"]') as HTMLInputElement
                                                        const h = hInput ? parseInt(hInput.value) || w : w
                                                        updatePending({ alert_min_area: w * h })
                                                    }}
                                                    data-dim="w"
                                                    className="w-20 bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-3 py-2 text-center text-slate-900 dark:text-white transition-colors" />
                                                <span className="text-slate-400 font-medium">×</span>
                                                <input type="number" min="1" max="200" step="1"
                                                    value={Math.round(Math.sqrt(pendingSettings?.alert_min_area || 625))}
                                                    onChange={e => {
                                                        const h = parseInt(e.target.value) || 25
                                                        const wInput = (e.target as HTMLInputElement).parentElement?.querySelector('[data-dim="w"]') as HTMLInputElement
                                                        const w = wInput ? parseInt(wInput.value) || h : h
                                                        updatePending({ alert_min_area: w * h })
                                                    }}
                                                    data-dim="h"
                                                    className="w-20 bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-3 py-2 text-center text-slate-900 dark:text-white transition-colors" />
                                                <span className="text-xs text-slate-400 ml-1">= {pendingSettings?.alert_min_area || 625} px²</span>
                                            </div>
                                        </div>
                                        </>)}
                                    </div>
                                </div>

                                {/* Features */}
                                <div>
                                    <h3 className="text-sm font-semibold text-cyan-600 dark:text-cyan-400 mb-3">Features</h3>
                                    <div className="grid grid-cols-2 gap-3">
                                        {appMode === 'single' && (
                                        <div>
                                            <label className="flex items-center gap-3 cursor-pointer">
                                                <input type="checkbox" checked={pendingSettings?.temporal_roi_enabled}
                                                    onChange={e => updatePending({ temporal_roi_enabled: e.target.checked })}
                                                    className="w-4 h-4 rounded border-slate-400 dark:border-slate-600 text-cyan-600 focus:ring-cyan-500" />
                                                <span className="text-sm text-slate-700 dark:text-slate-300">Temporal ROI</span>
                                            </label>
                                            <p className="text-xs text-slate-500 dark:text-slate-600 ml-7">Track detections across frames</p>
                                        </div>
                                        )}
                                        <div>
                                            <label className="flex items-center gap-3 cursor-pointer">
                                                <input type="checkbox" checked={pendingSettings?.save_video}
                                                    onChange={e => updatePending({ save_video: e.target.checked })}
                                                    className="w-4 h-4 rounded border-slate-400 dark:border-slate-600 text-cyan-600 focus:ring-cyan-500" />
                                                <span className="text-sm text-slate-700 dark:text-slate-300">Save Video</span>
                                            </label>
                                            <p className="text-xs text-slate-500 dark:text-slate-600 ml-7">Record annotated output</p>
                                        </div>
                                        {appMode === 'single' && (
                                        <div>
                                            <label className="flex items-center gap-3 cursor-pointer">
                                                <input type="checkbox" checked={pendingSettings?.save_alert_frames}
                                                    onChange={e => updatePending({ save_alert_frames: e.target.checked })}
                                                    className="w-4 h-4 rounded border-slate-400 dark:border-slate-600 text-cyan-600 focus:ring-cyan-500" />
                                                <span className="text-sm text-slate-700 dark:text-slate-300">Save Alert Frames</span>
                                            </label>
                                            <p className="text-xs text-slate-500 dark:text-slate-600 ml-7">Capture frames when alert triggers</p>
                                        </div>
                                        )}
                                    </div>
                                </div>

                                {/* Overlays */}
                                <div>
                                    <h3 className="text-sm font-semibold text-cyan-600 dark:text-cyan-400 mb-3">Overlays</h3>
                                    <div className="grid grid-cols-3 gap-3">
                                        <label className="flex items-center gap-3 cursor-pointer">
                                            <input type="checkbox" checked={pendingSettings?.show_gate}
                                                onChange={e => updatePending({ show_gate: e.target.checked })}
                                                className="w-4 h-4 rounded border-slate-400 dark:border-slate-600 text-cyan-600 focus:ring-cyan-500" />
                                            <span className="text-sm text-slate-700 dark:text-slate-300">Show Gate</span>
                                        </label>
                                        <label className="flex items-center gap-3 cursor-pointer">
                                            <input type="checkbox" checked={pendingSettings?.show_troi}
                                                onChange={e => updatePending({ show_troi: e.target.checked })}
                                                className="w-4 h-4 rounded border-slate-400 dark:border-slate-600 text-cyan-600 focus:ring-cyan-500" />
                                            <span className="text-sm text-slate-700 dark:text-slate-300">Show TROI</span>
                                        </label>
                                        {appMode === 'single' ? (
                                        <label className="flex items-center gap-3 cursor-pointer">
                                            <input type="checkbox" checked={pendingSettings?.show_cascade}
                                                onChange={e => updatePending({ show_cascade: e.target.checked })}
                                                className="w-4 h-4 rounded border-slate-400 dark:border-slate-600 text-cyan-600 focus:ring-cyan-500" />
                                            <span className="text-sm text-slate-700 dark:text-slate-300">Show Cascade</span>
                                        </label>
                                        ) : (
                                        <label className="flex items-center gap-3 cursor-pointer">
                                            <input type="checkbox" checked={pendingSettings?.show_cascade}
                                                onChange={e => updatePending({ show_cascade: e.target.checked })}
                                                className="w-4 h-4 rounded border-slate-400 dark:border-slate-600 text-cyan-600 focus:ring-cyan-500" />
                                            <span className="text-sm text-slate-700 dark:text-slate-300">Source Tags</span>
                                        </label>
                                        )}
                                    </div>
                                </div>

                                {/* Cooldowns */}
                                <div>
                                    <h3 className="text-sm font-semibold text-cyan-600 dark:text-cyan-400 mb-1">Cooldowns</h3>
                                    <p className="text-xs text-slate-500 dark:text-slate-600 mb-3">Warning = any detection. Alert = large drone confirmed.</p>
                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="text-xs text-slate-500 dark:text-slate-400">Warning Cooldown (s)</label>
                                            <p className="text-xs text-slate-500 dark:text-slate-600 mb-1">Pause after warning triggers</p>
                                            <input type="number" min="0" max="60" step="0.5" value={pendingSettings?.warning_cooldown}
                                                onChange={e => updatePending({ warning_cooldown: parseFloat(e.target.value) })}
                                                className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-3 py-2 text-slate-900 dark:text-white transition-colors" />
                                        </div>
                                        <div>
                                            <label className="text-xs text-slate-500 dark:text-slate-400">Alert Cooldown (s)</label>
                                            <p className="text-xs text-slate-500 dark:text-slate-600 mb-1">Pause after alert triggers</p>
                                            <input type="number" min="0" max="60" step="0.5" value={pendingSettings?.alert_cooldown}
                                                onChange={e => updatePending({ alert_cooldown: parseFloat(e.target.value) })}
                                                className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-3 py-2 text-slate-900 dark:text-white transition-colors" />
                                        </div>
                                    </div>
                                </div>

                                {/* Window Settings */}
                                <div>
                                    <h3 className="text-sm font-semibold text-cyan-600 dark:text-cyan-400 mb-1">Detection Windows (N-of-M)</h3>
                                    <p className="text-xs text-slate-500 dark:text-slate-600 mb-3">Require N hits in M frames to trigger</p>
                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="text-xs text-slate-500 dark:text-slate-400">Warning Window Size</label>
                                            <input type="number" min="3" max="30" value={pendingSettings?.warning_window_size}
                                                onChange={e => updatePending({ warning_window_size: parseInt(e.target.value) })}
                                                className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-3 py-2 text-slate-900 dark:text-white transition-colors" />
                                        </div>
                                        <div>
                                            <label className="text-xs text-slate-500 dark:text-slate-400">Warning Require Hits</label>
                                            <input type="number" min="1" max="30" value={pendingSettings?.warning_require_hits}
                                                onChange={e => updatePending({ warning_require_hits: parseInt(e.target.value) })}
                                                className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-3 py-2 text-slate-900 dark:text-white transition-colors" />
                                        </div>
                                        <div>
                                            <label className="text-xs text-slate-500 dark:text-slate-400">Alert Window Size</label>
                                            <input type="number" min="3" max="30" value={pendingSettings?.alert_window_size}
                                                onChange={e => updatePending({ alert_window_size: parseInt(e.target.value) })}
                                                className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-3 py-2 text-slate-900 dark:text-white transition-colors" />
                                        </div>
                                        <div>
                                            <label className="text-xs text-slate-500 dark:text-slate-400">Alert Require Hits</label>
                                            <input type="number" min="1" max="30" value={pendingSettings?.alert_require_hits}
                                                onChange={e => updatePending({ alert_require_hits: parseInt(e.target.value) })}
                                                className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-3 py-2 text-slate-900 dark:text-white transition-colors" />
                                        </div>
                                    </div>
                                    <div className="mt-3">
                                        <label className="text-xs text-slate-500 dark:text-slate-400">Alert Avg Conf Threshold</label>
                                        <p className="text-xs text-slate-500 dark:text-slate-600 mb-1">0 = disabled. Require avg confidence over hit frames ≥ this to fire alert</p>
                                        <input type="number" min="0" max="1" step="0.05" value={pendingSettings?.alert_avg_conf_threshold}
                                            onChange={e => updatePending({ alert_avg_conf_threshold: parseFloat(e.target.value) })}
                                            className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-3 py-2 text-slate-900 dark:text-white transition-colors" />
                                    </div>
                                </div>

                                {/* Log Mode - single mode only */}
                                {appMode === 'single' && (
                                <div>
                                    <h3 className="text-sm font-semibold text-cyan-600 dark:text-cyan-400 mb-3">Display</h3>
                                    <div>
                                        <label className="text-xs text-slate-500 dark:text-slate-400">Log Mode</label>
                                        <select value={pendingSettings?.log_mode}
                                            onChange={e => updatePending({ log_mode: e.target.value })}
                                            className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-400 dark:border-slate-700 rounded-lg px-3 py-2 mt-1 text-slate-900 dark:text-white transition-colors">
                                            <option value="off">Off</option>
                                            <option value="full">Full</option>
                                            <option value="windows_big">Windows Big</option>
                                        </select>
                                    </div>
                                </div>
                                )}
                            </div>

                            <div className="flex gap-3 mt-6">
                                <button onClick={() => setShowSettings(false)}
                                    className="flex-1 bg-slate-100 dark:bg-slate-700 py-3 rounded-xl font-semibold hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-700 dark:text-white transition-colors">
                                    Cancel
                                </button>
                                <button onClick={saveSettings}
                                    className="flex-1 bg-gradient-to-r from-emerald-500 to-green-600 py-3 rounded-xl font-semibold hover:from-emerald-400 hover:to-green-500 text-white shadow-lg shadow-emerald-900/20">
                                    Save Settings
                                </button>
                            </div>
                        </div>
                    </div>
                )}

            {/* Footer */}
            <footer className="mt-12 mb-6 text-center text-slate-400 dark:text-slate-500 text-xs tracking-wider space-y-1 transition-colors">
                <p>DEVELOPED BY</p>
                <div className="flex justify-center font-semibold text-slate-500 dark:text-slate-400">
                    <span>Ahmed Eltayeb - CS @UNISA</span>

                </div>
            </footer>
        </div >
    )
}
