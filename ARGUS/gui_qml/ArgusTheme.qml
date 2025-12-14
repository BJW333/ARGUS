pragma Singleton
import QtQuick 2.15

QtObject {
    // ══════════════════════════════════════════════════════
    // LIQUID GLASS DESIGN SYSTEM
    // Apple-inspired transparent glass UI tokens
    // ══════════════════════════════════════════════════════

    // ─────────────────────────────────────────────────────
    // Glass Base Properties
    // ─────────────────────────────────────────────────────
    readonly property real glassOpacity: 0.12
    readonly property real glassBorderOpacity: 0.35
    readonly property real glassBlurRadius: 64
    readonly property real glassHighlightOpacity: 0.4
    readonly property real glassShadowOpacity: 0.25

    // ─────────────────────────────────────────────────────
    // Glass Colors
    // ─────────────────────────────────────────────────────
    readonly property color glassTint: Qt.rgba(1, 1, 1, glassOpacity)
    readonly property color glassBorder: Qt.rgba(1, 1, 1, glassBorderOpacity)
    readonly property color glassHighlight: Qt.rgba(1, 1, 1, glassHighlightOpacity)
    readonly property color glassShadow: Qt.rgba(0, 0, 0, glassShadowOpacity)
    readonly property color glassInnerBorder: Qt.rgba(1, 1, 1, 0.08)

    // Hover states
    readonly property color glassHover: Qt.rgba(1, 1, 1, 0.18)
    readonly property color glassBorderHover: Qt.rgba(1, 1, 1, 0.5)

    // ─────────────────────────────────────────────────────
    // Text Colors
    // ─────────────────────────────────────────────────────
    readonly property color textPrimary: Qt.rgba(1, 1, 1, 0.98)
    readonly property color textSecondary: Qt.rgba(1, 1, 1, 0.7)
    readonly property color textTertiary: Qt.rgba(1, 1, 1, 0.5)
    readonly property color textMuted: Qt.rgba(1, 1, 1, 0.35)

    // ─────────────────────────────────────────────────────
    // Accent Colors
    // ─────────────────────────────────────────────────────
    readonly property color accentBlue: Qt.rgba(0.35, 0.65, 1.0, 1.0)
    readonly property color accentBlueDim: Qt.rgba(0.35, 0.65, 1.0, 0.3)
    readonly property color accentGold: Qt.rgba(1.0, 0.82, 0.4, 1.0)
    readonly property color accentGoldDim: Qt.rgba(1.0, 0.82, 0.4, 0.3)
    readonly property color accentGreen: Qt.rgba(0.2, 1.0, 0.55, 1.0)
    readonly property color accentRed: Qt.rgba(1.0, 0.35, 0.4, 1.0)

    // ─────────────────────────────────────────────────────
    // Neural Network / Brain Colors
    // ─────────────────────────────────────────────────────
    readonly property color neuralGoldBright: "#FFD700"
    readonly property color neuralGoldMid: "#FFA500"
    readonly property color neuralGoldLight: "#FFE4AA"
    readonly property color neuralAmber: "#FFBF00"
    readonly property color particleCore: "#FFE4AA"
    readonly property color particleGlow: "#FFC947"

    // ─────────────────────────────────────────────────────
    // Gradient Definitions
    // ─────────────────────────────────────────────────────
    readonly property var glassTopGradient: [
        { position: 0.0, color: Qt.rgba(1, 1, 1, 0.4) },
        { position: 0.25, color: Qt.rgba(1, 1, 1, 0.15) },
        { position: 0.5, color: Qt.rgba(1, 1, 1, 0.04) },
        { position: 1.0, color: "transparent" }
    ]

    readonly property var glassBottomGradient: [
        { position: 0.0, color: "transparent" },
        { position: 0.6, color: Qt.rgba(0, 0, 0, 0.08) },
        { position: 1.0, color: Qt.rgba(0, 0, 0, 0.25) }
    ]

    // ─────────────────────────────────────────────────────
    // Border Radius Values
    // ─────────────────────────────────────────────────────
    readonly property int radiusSmall: 12
    readonly property int radiusMedium: 18
    readonly property int radiusLarge: 24
    readonly property int radiusXLarge: 28

    // ─────────────────────────────────────────────────────
    // Shadow Definitions
    // ─────────────────────────────────────────────────────
    readonly property int shadowRadius: 35
    readonly property int shadowSamples: 28
    readonly property int shadowOffset: 10
    readonly property color shadowColor: Qt.rgba(0, 0, 0, 0.4)

    // ─────────────────────────────────────────────────────
    // Animation Durations
    // ─────────────────────────────────────────────────────
    readonly property int animFast: 100
    readonly property int animNormal: 180
    readonly property int animSlow: 300
    readonly property int animVerySlow: 500

    // ─────────────────────────────────────────────────────
    // Helper Functions
    // ─────────────────────────────────────────────────────
    function withAlpha(color, alpha) {
        return Qt.rgba(color.r, color.g, color.b, alpha)
    }

    function mixColors(color1, color2, ratio) {
        return Qt.rgba(
            color1.r * (1 - ratio) + color2.r * ratio,
            color1.g * (1 - ratio) + color2.g * ratio,
            color1.b * (1 - ratio) + color2.b * ratio,
            color1.a * (1 - ratio) + color2.a * ratio
        )
    }

    function glassColor(baseAlpha) {
        return Qt.rgba(1, 1, 1, baseAlpha || glassOpacity)
    }
}