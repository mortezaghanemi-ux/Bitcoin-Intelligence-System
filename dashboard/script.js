
"use strict";

/*
 * Bitcoin Intelligence System
 * Read-only dashboard frontend.
 *
 * API:
 * GET /api/dashboard
 *
 * No trading, order execution or data collection.
 */

const API_ENDPOINT = "/api/dashboard";

const DEFAULT_PAIR = "XBTUSD";
const DEFAULT_TIMEFRAME = "5m";

const MAX_CANDLES = 100;
const MAX_SIGNALS = 100;

const REFRESH_INTERVAL_MS = 60000;

const STATUS_LABELS = {
    AVAILABLE: "در دسترس",
    UNAVAILABLE: "ناموجود",
    STALE: "قدیمی",
    INVALID: "نامعتبر",
    SUCCESS: "موفق",
    ERROR: "خطا",
    COMPLETED: "تکمیل‌شده",
    DEGRADED: "دارای مشکل",
    PARTIAL: "ناقص",
    VALIDATED: "اعتبارسنجی‌شده",
    REQUIRES_MANUAL_REVIEW: "نیازمند بررسی دستی",
    LONG: "خرید",
    SHORT: "فروش",
    NO_TRADE: "بدون معامله",
    UNKNOWN: "نامشخص"
};

const ELEMENTS = {};

let refreshTimer = null;
let requestInProgress = false;
let lastSnapshot = null;


function getElement(id) {
    if (!ELEMENTS[id]) {
        ELEMENTS[id] = document.getElementById(id);
    }

    return ELEMENTS[id];
}


function setText(id, value) {
    const element = getElement(id);

    if (!element) {
        return;
    }

    element.textContent =
        value === null || value === undefined || value === ""
            ? "—"
            : String(value);
}


function formatNumber(value, maximumFractionDigits = 2) {
    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {
        return "—";
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return new Intl.NumberFormat("en-US", {
        maximumFractionDigits,
        minimumFractionDigits: 0
    }).format(number);
}


function formatTimestamp(value) {
    if (!value) {
        return "—";
    }

    const date = new Date(value);

    if (!Number.isFinite(date.getTime())) {
        return "نامعتبر";
    }

    return new Intl.DateTimeFormat("fa-IR", {
        dateStyle: "short",
        timeStyle: "medium",
        timeZone: "UTC"
    }).format(date) + " UTC";
}


function formatAge(value) {
    if (
        value === null ||
        value === undefined ||
        !Number.isFinite(Number(value))
    ) {
        return "نامشخص";
    }

    const minutes = Number(value);

    if (minutes < 60) {
        return `${formatNumber(minutes)} دقیقه`;
    }

    if (minutes < 1440) {
        return `${formatNumber(minutes / 60)} ساعت`;
    }

    return `${formatNumber(minutes / 1440)} روز`;
}


function translateStatus(status) {
    if (!status) {
        return STATUS_LABELS.UNKNOWN;
    }

    return STATUS_LABELS[status] || String(status);
}


function createElement(tag, className, text) {
    const element = document.createElement(tag);

    if (className) {
        element.className = className;
    }

    if (text !== undefined && text !== null) {
        element.textContent = String(text);
    }

    return element;
}


function clearElement(element) {
    if (element) {
        element.replaceChildren();
    }
}


function showError(message) {
    const container = getElement("error-container");
    const text = getElement("error-message");

    if (text) {
        text.textContent = message;
    }

    if (container) {
        container.hidden = false;
    }

    setConnectionStatus(false, "خطا در ارتباط");
}


function hideError() {
    const container = getElement("error-container");

    if (container) {
        container.hidden = true;
    }
}


function setConnectionStatus(connected, label) {
    const indicator = getElement("connection-indicator");

    setText(
        "connection-status",
        label || (connected ? "متصل" : "قطع ارتباط")
    );

    if (!indicator) {
        return;
    }

    indicator.style.backgroundColor =
        connected ? "var(--success)" : "var(--danger)";

    indicator.style.boxShadow = connected
        ? "0 0 10px rgba(37, 197, 138, 0.35)"
        : "0 0 10px rgba(239, 98, 98, 0.35)";
}


function setBadge(id, status) {
    const element = getElement(id);

    if (!element) {
        return;
    }

    const normalized = String(status || "UNKNOWN");

    element.textContent = translateStatus(normalized);

    const colors = {
        AVAILABLE: "var(--success)",
        SUCCESS: "var(--success)",
        VALIDATED: "var(--success)",
        STALE: "var(--warning)",
        PARTIAL: "var(--warning)",
        DEGRADED: "var(--warning)",
        UNAVAILABLE: "var(--danger)",
        INVALID: "var(--danger)",
        ERROR: "var(--danger)"
    };

    element.style.color =
        colors[normalized] || "var(--muted)";

    element.style.borderColor =
        colors[normalized] || "var(--border)";
}


function setOverviewStatus(status) {
    const element = getElement("overall-status");

    if (!element) {
        return;
    }

    element.textContent = translateStatus(status);

    const colors = {
        AVAILABLE: "var(--success)",
        PARTIAL: "var(--warning)",
        STALE: "var(--warning)",
        DEGRADED: "var(--danger)",
        INVALID: "var(--danger)"
    };

    element.style.color =
        colors[status] || "var(--muted)";
}


function renderCandles(marketData) {
    const body = getElement("candles-table-body");

    if (!body) {
        return;
    }

    clearElement(body);

    const candles = Array.isArray(marketData?.candles)
        ? marketData.candles
        : [];

    const latest = marketData?.latest_candle || null;

    if (latest) {
        setText("candle-open", formatNumber(latest.open, 2));
        setText("candle-high", formatNumber(latest.high, 2));
        setText("candle-low", formatNumber(latest.low, 2));
        setText("candle-close", formatNumber(latest.close, 2));
        setText("candle-volume", formatNumber(latest.volume, 4));

        setText(
            "current-price",
            formatNumber(latest.close, 2)
        );

        setText(
            "price-updated",
            formatTimestamp(latest.close_time)
        );
    } else {
        [
            "candle-open",
            "candle-high",
            "candle-low",
            "candle-close",
            "candle-volume",
            "current-price"
        ].forEach(id => setText(id, "—"));

        setText("price-updated", "اطلاعات قیمت موجود نیست");
    }

    setText(
        "market-data-status",
        translateStatus(marketData?.status)
    );

    setText(
        "market-data-age",
        `زمان سپری‌شده: ${formatAge(marketData?.age_minutes)}`
    );

    if (candles.length === 0) {
        const row = document.createElement("tr");
        const cell = createElement(
            "td",
            "empty-state",
            "کندل معتبری برای نمایش وجود ندارد."
        );

        cell.colSpan = 6;
        row.appendChild(cell);
        body.appendChild(row);
        return;
    }

    const visibleCandles = candles.slice(-MAX_CANDLES).reverse();

    for (const candle of visibleCandles) {
        const row = document.createElement("tr");

        const values = [
            formatTimestamp(candle.open_time),
            formatNumber(candle.open, 2),
            formatNumber(candle.high, 2),
            formatNumber(candle.low, 2),
            formatNumber(candle.close, 2),
            formatNumber(candle.volume, 4)
        ];

        for (const value of values) {
            row.appendChild(createElement("td", "", value));
        }

        body.appendChild(row);
    }
}


function renderObject(container, data, depth = 0) {
    if (!container) {
        return;
    }

    if (data === null || data === undefined) {
        container.appendChild(
            createElement(
                "p",
                "empty-state",
                "اطلاعاتی برای نمایش موجود نیست."
            )
        );
        return;
    }

    if (depth > 3) {
        container.appendChild(
            createElement("p", "empty-state", "جزئیات بیشتر موجود است.")
        );
        return;
    }

    if (Array.isArray(data)) {
        if (data.length === 0) {
            container.appendChild(
                createElement("p", "empty-state", "موردی ثبت نشده است.")
            );
            return;
        }

        const list = document.createElement("ul");

        for (const item of data.slice(0, 30)) {
            const listItem = document.createElement("li");

            if (
                item !== null &&
                typeof item === "object"
            ) {
                renderObject(listItem, item, depth + 1);
            } else {
                listItem.textContent = formatValue(item);
            }

            list.appendChild(listItem);
        }

        container.appendChild(list);
        return;
    }

    if (typeof data !== "object") {
        container.appendChild(
            createElement("p", "", formatValue(data))
        );
        return;
    }

    const grid = createElement("div", "dashboard-data-grid");

    const entries = Object.entries(data).slice(0, 40);

    for (const [key, value] of entries) {
        if (
            value === null ||
            value === undefined ||
            typeof value === "object"
        ) {
            if (Array.isArray(value) && value.length === 0) {
                continue;
            }

            if (
                !Array.isArray(value) &&
                value !== null &&
                typeof value === "object" &&
                Object.keys(value).length === 0
            ) {
                continue;
            }
        }

        const item = createElement("div", "dashboard-data-item");

        item.appendChild(
            createElement("span", "small-label", translateKey(key))
        );

        const valueElement = document.createElement("strong");

        if (
            value !== null &&
            typeof value === "object"
        ) {
            renderObject(valueElement, value, depth + 1);
        } else {
            valueElement.textContent = formatValue(value);
        }

        item.appendChild(valueElement);
        grid.appendChild(item);
    }

    if (entries.length === 0) {
        container.appendChild(
            createElement("p", "empty-state", "اطلاعاتی موجود نیست.")
        );
        return;
    }

    container.appendChild(grid);
}


function translateKey(key) {
    const labels = {
        asset: "دارایی",
        pair: "جفت‌ارز",
        timeframe: "بازه زمانی",
        direction: "جهت",
        signal: "سیگنال",
        status: "وضعیت",
        confidence_score: "امتیاز اطمینان",
        confidence: "اطمینان",
        market_price: "قیمت بازار",
        current_price: "قیمت فعلی",
        entry: "قیمت ورود",
        entry_price: "قیمت ورود",
        entry_range: "محدوده ورود",
        stop_loss: "حد ضرر",
        take_profit: "حد سود",
        risk_reward: "نسبت ریسک به بازده",
        risk_reward_ratio: "نسبت ریسک به بازده",
        leverage: "اهرم",
        position_size: "اندازه موقعیت",
        position_value: "ارزش موقعیت",
        risk_amount: "مقدار ریسک",
        analysis_timestamp: "زمان تحلیل",
        generated_at: "زمان تولید گزارش",
        forecast_timestamp: "زمان پیش‌بینی",
        forecast_horizon: "افق پیش‌بینی",
        market_condition: "شرایط بازار",
        probabilities: "احتمالات",
        influencing_factors: "عوامل تأثیرگذار",
        risk_factors: "عوامل ریسک",
        limitations: "محدودیت‌ها",
        reasons: "دلایل",
        rejection_reasons: "دلایل رد",
        source: "منبع",
        sources: "منابع",
        open: "باز شدن",
        high: "بالاترین",
        low: "پایین‌ترین",
        close: "بسته شدن",
        volume: "حجم",
        open_time: "زمان باز شدن",
        close_time: "زمان بسته شدن",
        age_minutes: "سن داده به دقیقه",
        records_returned: "تعداد رکوردها",
        manual_review_only: "فقط بررسی دستی",
        auto_trading: "معاملات خودکار",
        exchange_connection: "اتصال صرافی",
        order_execution: "اجرای سفارش",
        net_profit: "سود خالص",
        net_return_pct: "بازده خالص",
        win_rate_pct: "درصد معاملات موفق",
        total_trades: "تعداد کل معاملات",
        maximum_drawdown_pct: "حداکثر افت سرمایه",
        total_fees: "مجموع کارمزد",
        total_pnl: "سود و زیان",
        model: "مدل",
        regime: "رژیم بازار",
        trend_strength: "قدرت روند",
        alignment: "هم‌جهتی",
        ratio: "نسبت",
        score: "امتیاز",
        name: "نام",
        description: "توضیحات",
        message: "پیام"
    };

    return labels[key] || key.replaceAll("_", " ");
}


function formatValue(value) {
    if (value === null || value === undefined) {
        return "—";
    }

    if (typeof value === "boolean") {
        return value ? "بله" : "خیر";
    }

    if (typeof value === "number") {
        return formatNumber(value, 4);
    }

    if (typeof value === "string") {
        if (
            /^\d{4}-\d{2}-\d{2}T/.test(value)
        ) {
            return formatTimestamp(value);
        }

        return translateStatus(value);
    }

    return String(value);
}


function renderReportSection(
    contentId,
    badgeId,
    section
) {
    const container = getElement(contentId);

    if (!container) {
        return;
    }

    clearElement(container);

    setBadge(
        badgeId,
        section?.status || "UNAVAILABLE"
    );

    if (!section || !section.data) {
        const message = section?.status === "STALE"
            ? "گزارش موجود است اما قدیمی شده است."
            : "گزارش معتبر و قابل نمایش موجود نیست.";

        container.appendChild(
            createElement("p", "empty-state", message)
        );

        return;
    }

    if (section.generated_at) {
        container.appendChild(
            createElement(
                "p",
                "small-label",
                `زمان گزارش: ${formatTimestamp(section.generated_at)}`
            )
        );
    }

    renderObject(container, section.data);
}


function isDisplayableSignal(signal) {
    if (!signal || typeof signal !== "object") {
        return false;
    }

    if (!["LONG", "SHORT"].includes(signal.signal)) {
        return false;
    }

    if (signal.status !== "REQUIRES_MANUAL_REVIEW") {
        return false;
    }

    const execution = signal.execution;
    const quality = signal.data_quality;

    if (!execution || !quality) {
        return false;
    }

    return (
        execution.manual_review_only === true &&
        execution.auto_trading === false &&
        execution.exchange_connection === false &&
        execution.order_execution === false &&
        quality.status === "VALIDATED" &&
        quality.closed_candles_only === true &&
        quality.future_data_used === false
    );
}


function renderTrading(trading) {
    const container = getElement("trading-content");

    if (!container) {
        return;
    }

    clearElement(container);

    const signals = Array.isArray(trading?.signals)
        ? trading.signals.filter(isDisplayableSignal)
        : [];

    setBadge(
        "trading-status",
        trading?.status || "UNAVAILABLE"
    );

    setText("signal-count", formatNumber(signals.length, 0));

    if (signals.length === 0) {
        container.appendChild(
            createElement(
                "p",
                "empty-state",
                "سیگنال معتبر و نیازمند بررسی دستی وجود ندارد."
            )
        );
        return;
    }

    for (const signal of signals.slice(0, MAX_SIGNALS)) {
        const card = createElement("article", "signal-card");

        const header = createElement("div", "signal-card-header");

        const direction = signal.signal === "LONG"
            ? "خرید (LONG)"
            : "فروش (SHORT)";

        const title = createElement("strong", "", direction);

        title.style.color = signal.signal === "LONG"
            ? "var(--success)"
            : "var(--danger)";

        header.appendChild(title);

        header.appendChild(
            createElement(
                "span",
                "status-badge",
                "بررسی دستی"
            )
        );

        card.appendChild(header);

        const details = document.createElement("div");

        renderObject(details, {
            analysis_timestamp: signal.analysis_timestamp,
            current_price: signal.current_price,
            timeframe: signal.timeframe,
            entry_range: signal.entry_range,
            stop_loss: signal.stop_loss,
            take_profit: signal.take_profit,
            risk_reward: signal.risk_reward,
            confidence: signal.confidence,
            reasons: signal.reasons
        });

        card.appendChild(details);
        container.appendChild(card);
    }
}


function renderSystemStatus(statuses) {
    const container = getElement("system-status-list");

    if (!container) {
        return;
    }

    clearElement(container);

    if (
        !statuses ||
        typeof statuses !== "object"
    ) {
        container.appendChild(
            createElement(
                "p",
                "empty-state",
                "وضعیت اجزای سیستم موجود نیست."
            )
        );
        return;
    }

    for (const [name, status] of Object.entries(statuses)) {
        const row = document.createElement("div");

        row.appendChild(
            createElement("span", "", translateKey(name))
        );

        const value = createElement(
            "strong",
            "",
            translateStatus(status)
        );

        const colors = {
            AVAILABLE: "var(--success)",
            STALE: "var(--warning)",
            UNAVAILABLE: "var(--danger)",
            INVALID: "var(--danger)"
        };

        value.style.color = colors[status] || "var(--muted)";

        row.appendChild(value);
        container.appendChild(row);
    }
}


function renderLimitations(limitations) {
    const container = getElement("limitations-list");

    if (!container) {
        return;
    }

    clearElement(container);

    const items = Array.isArray(limitations)
        ? limitations
        : [];

    if (items.length === 0) {
        container.appendChild(
            createElement("li", "", "محدودیتی برای نمایش ثبت نشده است.")
        );
        return;
    }

    for (const item of items.slice(0, 30)) {
        container.appendChild(
            createElement("li", "", item)
        );
    }
}


function renderDashboard(snapshot) {
    if (
        !snapshot ||
        typeof snapshot !== "object" ||
        snapshot.schema_version !== 1
    ) {
        throw new Error("ساختار پاسخ سرور معتبر نیست.");
    }

    lastSnapshot = snapshot;

    setConnectionStatus(true, "متصل به سرور");

    setOverviewStatus(snapshot.status);

    setText(
        "last-updated",
        `آخرین به‌روزرسانی: ${formatTimestamp(snapshot.generated_at)}`
    );

    renderCandles(snapshot.market_data || {});

    renderReportSection(
        "market-content",
        "market-status",
        snapshot.market
    );

    renderReportSection(
        "forecast-content",
        "forecast-status",
        snapshot.forecast
    );

    renderTrading(snapshot.trading || {});

    renderReportSection(
        "quality-content",
        "quality-status",
        snapshot.data_quality
    );

    renderSystemStatus(snapshot.system_status);

    renderLimitations(snapshot.limitations);
}


async function loadDashboard() {
    if (requestInProgress) {
        return;
    }

    requestInProgress = true;

    const button = getElement("refresh-button");

    if (button) {
        button.disabled = true;
        button.textContent = "در حال دریافت...";
    }

    try {
        hideError();

        const pairElement = getElement("pair-select");
        const timeframeElement = getElement("timeframe-select");

        const pair = pairElement?.value || DEFAULT_PAIR;
        const timeframe = timeframeElement?.value || DEFAULT_TIMEFRAME;

        const parameters = new URLSearchParams({
            pair,
            timeframe,
            candle_limit: String(MAX_CANDLES),
            signal_limit: String(MAX_SIGNALS)
        });

        const response = await fetch(
            `${API_ENDPOINT}?${parameters.toString()}`,
            {
                method: "GET",
                headers: {
                    "Accept": "application/json"
                },
                cache: "no-store"
            }
        );

        if (!response.ok) {
            let message = `خطای سرور: ${response.status}`;

            try {
                const errorData = await response.json();

                if (errorData.error) {
                    message = errorData.error;
                }
            } catch {
                // Keep the HTTP status message.
            }

            throw new Error(message);
        }

        const snapshot = await response.json();

        renderDashboard(snapshot);

    } catch (error) {
        const message = error instanceof Error
            ? error.message
            : "خطای ناشناخته هنگام دریافت اطلاعات.";

        showError(message);

    } finally {
        requestInProgress = false;

        if (button) {
            button.disabled = false;
            button.textContent = "به‌روزرسانی";
        }
    }
}


function initializeDashboard() {
    const refreshButton = getElement("refresh-button");
    const pairSelect = getElement("pair-select");
    const timeframeSelect = getElement("timeframe-select");

    if (refreshButton) {
        refreshButton.addEventListener(
            "click",
            loadDashboard
        );
    }

    if (pairSelect) {
        pairSelect.addEventListener(
            "change",
            loadDashboard
        );
    }

    if (timeframeSelect) {
        timeframeSelect.addEventListener(
            "change",
            loadDashboard
        );
    }

    setConnectionStatus(false, "در انتظار اتصال");

    loadDashboard();

    if (refreshTimer !== null) {
        clearInterval(refreshTimer);
    }

    refreshTimer = setInterval(
        loadDashboard,
        REFRESH_INTERVAL_MS
    );
}


if (document.readyState === "loading") {
    document.addEventListener(
        "DOMContentLoaded",
        initializeDashboard,
        { once: true }
    );
} else {
    initializeDashboard();
}
