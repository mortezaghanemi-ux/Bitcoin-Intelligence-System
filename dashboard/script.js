
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

// Published copy (GitHub Pages): one JSON file per timeframe.
const STATIC_MODE = Boolean(
    document.querySelector('meta[name="dashboard-static"]')
);

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


function formatPercentValue(value) {
    if (value === null || value === undefined) {
        return "—";
    }

    return `${formatNumber(value, 1)}٪`;
}


function translateScope(scope) {
    if (!scope) {
        return "—";
    }

    if (scope === "all signals") {
        return "همه سیگنال‌ها";
    }

    if (scope === "LONG signals") {
        return "سیگنال‌های خرید";
    }

    if (scope === "SHORT signals") {
        return "سیگنال‌های فروش";
    }

    if (scope.startsWith("confidence ")) {
        return `اطمینان ${scope.slice("confidence ".length)}`;
    }

    return scope;
}


function describeHistoricalPerformance(performance) {
    if (!performance || performance.status === "NO_BACKTEST") {
        return {
            text: "وین‌ریت گذشته: هنوز بک‌تست اجرا نشده است.",
            warning: true
        };
    }

    const verdictNote = describeEdgeVerdict(performance);

    const interval = Array.isArray(performance.win_rate_ci95_pct)
        ? ` (بازه اطمینان ۹۵٪: ${formatNumber(performance.win_rate_ci95_pct[0], 0)} تا ` +
          `${formatNumber(performance.win_rate_ci95_pct[1], 0)})`
        : "";

    const text =
        `وین‌ریت اندازه‌گیری‌شده در گذشته: ` +
        `${formatPercentValue(performance.win_rate_pct)}${interval} ` +
        `از ${formatNumber(performance.trades, 0)} معامله شبیه‌سازی‌شده ` +
        `(${translateScope(performance.scope)}، هدف ${performance.target || "TP1"}).`;

    return {
        text,
        warning:
            performance.status !== "MEASURED" ||
            performance.edge_verdict === "NO_EDGE_MEASURED",
        note: verdictNote ||
            (performance.status !== "MEASURED"
                ? "تعداد نمونه کم است؛ این عدد قابل اتکا نیست."
                : "این عدد اندازه‌گیری گذشته است، نه تضمین آینده.")
    };
}


const LIVE_STATUS_TEXT = {
    INSUFFICIENT_SAMPLE: "نمونه زنده هنوز کافی نیست",
    CONSISTENT: "نتیجه زنده با بک‌تست سازگار است",
    UNDERPERFORMING: "عملکرد زنده ضعیف است؛ سیگنال‌ها متوقف‌شده فرض شوند",
    WORSE_THAN_BACKTEST: "عملکرد زنده بدتر از بک‌تست است؛ سیگنال‌ها متوقف‌شده فرض شوند"
};

function renderLivePerformance(snapshot) {
    const container = document.getElementById("performance-content");

    if (!container) {
        return;
    }

    const gate = snapshot.performance?.report?.activation_gate;

    if (gate) {
        const gateLine = createElement(
            "p",
            "",
            gate.passed
                ? "شرط فعال‌سازی: قبول (مزیت در بک‌تست اندازه‌گیری شد)"
                : "شرط فعال‌سازی: رد (مزیت قابل‌اتکا در بک‌تست ثابت نشد)"
        );
        gateLine.style.fontWeight = "700";
        gateLine.style.color = gate.passed
            ? "var(--success)"
            : "var(--danger)";
        container.appendChild(gateLine);
    }

    const live = snapshot.live_performance;

    if (!live || live.status !== "AVAILABLE") {
        container.appendChild(
            createElement(
                "p",
                "empty-state",
                "هنوز نتیجه زنده‌ای برای سیگنال‌های صادرشده ثبت نشده است."
            )
        );
        return;
    }

    const summary = live.summary || {};
    const comparison = live.comparison || {};
    const profit = summary.profit_factor === null ||
        summary.profit_factor === undefined
        ? "—"
        : formatNumber(summary.profit_factor, 2);

    container.appendChild(
        createElement(
            "h3",
            "",
            "نتیجه سیگنال‌های واقعی (پس از صدور)"
        )
    );

    const line = createElement(
        "p",
        "",
        `تمام‌شده: ${summary.resolved} (برد ${summary.wins}، باخت ${summary.losses})، ` +
        `باز: ${summary.open}، درصد برد: ` +
        `${summary.win_rate_pct === null ? "—" : formatNumber(summary.win_rate_pct, 1) + "٪"}، ` +
        `ضریب سود: ${profit}`
    );
    container.appendChild(line);

    const status = createElement(
        "p",
        "",
        LIVE_STATUS_TEXT[comparison.status] || comparison.message || ""
    );
    status.style.color = comparison.signals_paused
        ? "var(--danger)"
        : "var(--warning)";
    status.style.fontWeight = "700";
    container.appendChild(status);

    container.appendChild(
        createElement(
            "p",
            "empty-state",
            "کارمزد و لغزش قیمت در این پیگیری لحاظ نشده؛ نتیجه کمی خوش‌بینانه‌تر از واقعیت است."
        )
    );
}

let livePaused = false;

function describeEdgeVerdict(performance) {
    if (performance?.edge_verdict !== "NO_EDGE_MEASURED") {
        return "";
    }

    return (
        "هشدار: در بک‌تست گذشته، این قوانین سودده نبوده‌اند " +
        `(ضریب سود ${formatNumber(performance.overall_profit_factor, 2)}، ` +
        `بازده کل ${formatPercentValue(performance.total_return_pct)}، ` +
        `بیشینه افت ${formatPercentValue(performance.maximum_drawdown_pct)}). ` +
        "درصد برد به‌تنهایی نشانه سودآوری نیست."
    );
}


function renderPerformanceTable(report) {
    const wrapper = createElement("div", "table-wrapper");
    const table = document.createElement("table");
    const head = document.createElement("thead");
    const headRow = document.createElement("tr");

    for (const label of [
        "گروه",
        "تعداد معامله",
        "وین‌ریت",
        "ضریب سود",
        "بازه اطمینان ۹۵٪",
        "وضعیت نمونه"
    ]) {
        headRow.appendChild(createElement("th", "", label));
    }

    head.appendChild(headRow);
    table.appendChild(head);

    const body = document.createElement("tbody");

    const rows = [
        ["همه سیگنال‌ها", report.overall],
        ["خارج از نمونه (بخش آخر دوره)", report.out_of_sample],
        ["خرید (LONG)", report.by_direction?.LONG],
        ["فروش (SHORT)", report.by_direction?.SHORT]
    ];

    for (const [name, group] of Object.entries(
        report.by_confidence || {}
    )) {
        rows.push([`اطمینان ${name}`, group]);
    }

    for (const [label, group] of rows) {
        if (!group) {
            continue;
        }

        const row = document.createElement("tr");
        const interval = Array.isArray(group.win_rate_ci95_pct)
            ? `${formatNumber(group.win_rate_ci95_pct[0], 0)} تا ` +
              `${formatNumber(group.win_rate_ci95_pct[1], 0)}`
            : "—";

        row.appendChild(createElement("td", "", label));
        row.appendChild(
            createElement("td", "", formatNumber(group.trades, 0))
        );
        row.appendChild(
            createElement("td", "", formatPercentValue(group.win_rate_pct))
        );
        row.appendChild(
            createElement(
                "td",
                "",
                group.profit_factor === null ||
                group.profit_factor === undefined
                    ? "—"
                    : formatNumber(group.profit_factor, 2)
            )
        );
        row.appendChild(createElement("td", "", interval));
        row.appendChild(
            createElement(
                "td",
                "",
                group.sample_status === "SUFFICIENT"
                    ? "کافی"
                    : "نمونه کم"
            )
        );

        body.appendChild(row);
    }

    table.appendChild(body);
    wrapper.appendChild(table);

    return wrapper;
}


function renderPerformance(performance) {
    const container = getElement("performance-content");

    if (!container) {
        return;
    }

    clearElement(container);

    setBadge(
        "performance-status",
        performance?.status || "UNAVAILABLE"
    );

    const report = performance?.report;

    if (!report) {
        container.appendChild(
            createElement(
                "p",
                "empty-state",
                "هنوز بک‌تستی اجرا نشده است. دستور backtest را اجرا کنید."
            )
        );
        return;
    }

    const period = report.period || {};

    const reportVerdict = describeEdgeVerdict({
        edge_verdict: report.overall?.profit_factor < 1 ||
            report.out_of_sample?.profit_factor < 1
            ? "NO_EDGE_MEASURED"
            : "OTHER",
        overall_profit_factor: report.overall?.profit_factor,
        total_return_pct: report.risk?.total_return_pct,
        maximum_drawdown_pct: report.risk?.maximum_drawdown_pct
    });

    if (reportVerdict) {
        const banner = createElement("p", "", reportVerdict);
        banner.style.color = "var(--danger)";
        banner.style.fontWeight = "700";
        container.appendChild(banner);
    }

    container.appendChild(
        createElement(
            "p",
            "",
            `دوره: ${formatNumber(period.days, 1)} روز · ` +
            `هدف سود: ${report.target || "TP1"} · ` +
            `بیشینه افت سرمایه: ${formatPercentValue(
                report.risk?.maximum_drawdown_pct
            )}`
        )
    );

    container.appendChild(renderPerformanceTable(report));

    container.appendChild(
        createElement(
            "p",
            "empty-state",
            "شبیه‌سازی روی داده گذشته با کارمزد و slippage؛ " +
            "پیش‌بینی آینده نیست و نتیجه واقعی می‌تواند متفاوت باشد."
        )
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

    const latestPrice = Number(
        lastSnapshot?.market_data?.latest_candle?.close
    );

    signals.slice(0, MAX_SIGNALS).forEach((signal, index) => {
        container.appendChild(
            buildSignalCard(
                signal,
                index === 0 && Number.isFinite(latestPrice)
                    ? latestPrice
                    : null
            )
        );
    });
}

function describeSignalLiveStatus(signal, price) {
    const isLong = signal.signal === "LONG";
    const stop = Number(signal.stop_loss);
    const first = Number(signal.take_profit?.[0]?.price);
    const lower = Number(signal.entry_range?.lower);
    const upper = Number(signal.entry_range?.upper);

    if (price === null || !Number.isFinite(stop)) {
        return { text: "قدیمی (سیگنال قبلی)", tone: "muted" };
    }

    if (isLong ? price <= stop : price >= stop) {
        return { text: "باطل شد: قیمت به حد ضرر رسید", tone: "danger" };
    }

    if (Number.isFinite(first) && (isLong ? price >= first : price <= first)) {
        return { text: "قیمت به حد سود اول رسید", tone: "success" };
    }

    if (
        Number.isFinite(lower) &&
        Number.isFinite(upper) &&
        price >= Math.min(lower, upper) &&
        price <= Math.max(lower, upper)
    ) {
        return { text: "قیمت در محدوده ورود است", tone: "success" };
    }

    return {
        text: "قیمت از محدوده ورود فاصله گرفته؛ ورود دیرهنگام است",
        tone: "warning"
    };
}

function buildSignalCard(signal, livePrice) {
    const card = createElement("article", "signal-card");
    const isLong = signal.signal === "LONG";
    const dollars = (value) =>
        Number.isFinite(Number(value))
            ? `${formatNumber(Number(value), 2)} دلار`
            : "—";

    const range = signal.entry_range || {};
    const stopValue = Number(signal.stop_loss);
    const currentPrice = Number(signal.current_price);
    const riskPercent =
        Number.isFinite(stopValue) && currentPrice > 0
            ? (Math.abs(currentPrice - stopValue) / currentPrice) * 100
            : null;

    const live = describeSignalLiveStatus(signal, livePrice);

    if (livePaused) {
        const paused = createElement(
            "p",
            "",
            "هشدار: عملکرد زنده این سیستم ضعیف بوده؛ سیگنال را اجرا نکنید."
        );
        paused.style.color = "var(--danger)";
        paused.style.fontWeight = "700";
        card.appendChild(paused);
    }

    const rows = [
        ["جهت", `${isLong ? "🟢 LONG" : "🔴 SHORT"}`],
        ["تایم‌فریم اجرا", signal.timeframe || "—"],
        [
            "محدوده ورود",
            Number.isFinite(Number(range.lower)) &&
            Number.isFinite(Number(range.upper))
                ? `${formatNumber(Math.min(range.lower, range.upper), 2)}` +
                  ` – ${formatNumber(Math.max(range.lower, range.upper), 2)} دلار`
                : "—"
        ],
        ["قیمت مرجع هنگام تولید", dollars(signal.current_price)],
        ["حد ضرر", dollars(signal.stop_loss)]
    ];

    (signal.take_profit || []).forEach((level) => {
        rows.push([`حد سود ${toPersianLevel(level.level)}`, dollars(level.price)]);
    });

    rows.push(
        ["ریسک به بازده", formatNumber(Number(signal.risk_reward), 2)],
        [
            "فاصله حد ضرر از قیمت",
            riskPercent === null ? "—" : `${formatNumber(riskPercent, 2)}٪`
        ],
        ["اطمینان الگوریتمی", formatNumber(Number(signal.confidence), 0)],
        ["زمان تحلیل", formatTimestamp(signal.analysis_timestamp)],
        ["وضعیت فعلی", live.text]
    );

    const table = createElement("table", "signal-table");
    const body = document.createElement("tbody");

    const head = document.createElement("tr");
    head.appendChild(createElement("th", "", "پارامتر"));
    head.appendChild(createElement("th", "", "مقدار"));
    const thead = document.createElement("thead");
    thead.appendChild(head);
    table.appendChild(thead);

    for (const [label, value] of rows) {
        const row = document.createElement("tr");
        row.appendChild(createElement("td", "", label));
        const cell = createElement("td", "signal-value", String(value));
        if (label === "وضعیت فعلی") {
            cell.dataset.tone = live.tone;
        }
        row.appendChild(cell);
        body.appendChild(row);
    }

    table.appendChild(body);
    card.appendChild(table);

    if (Array.isArray(signal.reasons) && signal.reasons.length > 0) {
        const reasons = createElement("ul", "signal-reasons");
        for (const reason of signal.reasons) {
            reasons.appendChild(createElement("li", "", String(reason)));
        }
        card.appendChild(reasons);
    }

    const history = describeHistoricalPerformance(
        signal.historical_performance
    );

    const historyLine = createElement("p", "", history.text);
    historyLine.style.color = history.warning
        ? "var(--warning)"
        : "var(--success)";
    card.appendChild(historyLine);

    if (history.note) {
        card.appendChild(createElement("p", "empty-state", history.note));
    }

    card.appendChild(
        createElement(
            "p",
            "empty-state",
            "اطمینان الگوریتمی امتیاز قوانین است، نه احتمال برد. " +
            "این سیگنال فقط برای بررسی دستی است و معامله خودکار انجام نمی‌شود."
        )
    );

    return card;
}

function toPersianLevel(level) {
    return ["اول", "دوم", "سوم"][Number(level) - 1] || String(level);
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

    livePaused = Boolean(
        snapshot.live_performance?.comparison?.signals_paused
    );
    renderTrading(snapshot.trading || {});

    renderPerformance(snapshot.performance || {});
    renderLivePerformance(snapshot);

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

        const requestUrl = STATIC_MODE
            ? `data/${timeframe}.json`
            : `${API_ENDPOINT}?${parameters.toString()}`;

        const response = await fetch(
            requestUrl,
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
