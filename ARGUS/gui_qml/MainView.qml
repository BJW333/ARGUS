import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "."

ApplicationWindow {
    id: window
    visible: true
    width: 1280
    height: 720
    title: "ARGUS"
    color: "transparent"
    flags: Qt.FramelessWindowHint | Qt.Window

    // ═══════════════════════════════════════════════════════
    // LIQUID GLASS DESIGN TOKENS
    // ═══════════════════════════════════════════════════════
    readonly property real glassOpacity: 0.12
    readonly property real glassBorderOpacity: 0.35
    readonly property color glassTint: Qt.rgba(1, 1, 1, glassOpacity)
    readonly property color glassBorder: Qt.rgba(1, 1, 1, glassBorderOpacity)
    readonly property color glassHighlight: Qt.rgba(1, 1, 1, 0.4)
    readonly property color glassShadow: Qt.rgba(0, 0, 0, 0.25)

    // ═══════════════════════════════════════════════════════
    // LIVE DATA PROPERTIES (updated by timer)
    // ═══════════════════════════════════════════════════════
    property string cpuLoad: "0%"
    property bool networkOnline: false

    Timer {
        interval: 4000  // 4 seconds - reduced CPU polling
        running: true
        repeat: true
        onTriggered: {
            cpuLoad = Backend.getCpuLoad()
            networkOnline = Backend.isNetworkConnected()
        }
    }

    Component.onCompleted: {
        cpuLoad = Backend.getCpuLoad()
        networkOnline = Backend.isNetworkConnected()
    }

    // ═══════════════════════════════════════════════════════
    // WINDOW DRAG AREA (title bar replacement)
    // ═══════════════════════════════════════════════════════
    MouseArea {
        id: dragArea
        anchors.fill: parent
        anchors.bottomMargin: parent.height - 50
        property point lastPos
        onPressed: lastPos = Qt.point(mouseX, mouseY)
        onPositionChanged: {
            if (pressed) {
                var dx = mouseX - lastPos.x
                var dy = mouseY - lastPos.y
                window.x += dx
                window.y += dy
            }
        }
    }

    // ═══════════════════════════════════════════════════════
    // AMBIENT BACKDROP (simplified - no shader)
    // ═══════════════════════════════════════════════════════
    Rectangle {
        anchors.centerIn: parent
        width: parent.width * 1.2
        height: parent.height * 1.2
        radius: width / 2
        color: Qt.rgba(0.3, 0.5, 0.9, 0.08)
        opacity: 0.4
    }

    // ═══════════════════════════════════════════════════════
    // MAIN CONTENT LAYOUT
    // ═══════════════════════════════════════════════════════
    property bool cardsCollapsed: false

    Row {
        id: mainRow
        anchors.fill: parent
        anchors.margins: 20
        spacing: 16

        // ───────────────────────────────────────────────────
        // LEFT SIDE: BRAIN + CHAT
        // ───────────────────────────────────────────────────
        Column {
            id: leftColumn
            width: cardsCollapsed ? parent.width - 70 : parent.width * 0.72
            height: parent.height
            spacing: 16

            Behavior on width {
                NumberAnimation { duration: 300; easing.type: Easing.InOutQuad }
            }

            // NEURAL BRAIN PANEL
            Item {
                id: topPanel
                width: parent.width
                height: parent.height * 0.38

                Rectangle {
                    id: brainGlass
                    anchors.fill: parent
                    radius: 28
                    color: glassTint
                    border.width: 1
                    border.color: glassBorder

                    // Top specular highlight
                    Rectangle {
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.margins: 1
                        height: parent.height * 0.5
                        radius: 27
                        gradient: Gradient {
                            GradientStop { position: 0.0; color: glassHighlight }
                            GradientStop { position: 0.3; color: Qt.rgba(1,1,1,0.15) }
                            GradientStop { position: 0.6; color: Qt.rgba(1,1,1,0.03) }
                            GradientStop { position: 1.0; color: "transparent" }
                        }
                    }

                    // Bottom shadow
                    Rectangle {
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.bottom: parent.bottom
                        anchors.margins: 1
                        height: parent.height * 0.3
                        radius: 27
                        gradient: Gradient {
                            GradientStop { position: 0.0; color: "transparent" }
                            GradientStop { position: 0.7; color: Qt.rgba(0,0,0,0.08) }
                            GradientStop { position: 1.0; color: glassShadow }
                        }
                    }

                    // Inner border
                    Rectangle {
                        anchors.fill: parent
                        anchors.margins: 1
                        radius: 27
                        color: "transparent"
                        border.width: 1
                        border.color: Qt.rgba(1, 1, 1, 0.08)
                    }
                }

                BrainCanvas {
                    id: brainCanvas
                    anchors.fill: parent
                    anchors.margins: 8
                    audioLevel: Backend.audioLevel
                }
            }

            // CHAT PANEL
            Item {
                id: chatContainer
                width: parent.width
                height: parent.height * 0.55

                Rectangle {
                    id: chatGlass
                    anchors.fill: parent
                    radius: 26
                    color: glassTint
                    border.width: 1
                    border.color: glassBorder

                    Rectangle {
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.margins: 1
                        height: parent.height * 0.4
                        radius: 25
                        gradient: Gradient {
                            GradientStop { position: 0.0; color: glassHighlight }
                            GradientStop { position: 0.25; color: Qt.rgba(1,1,1,0.12) }
                            GradientStop { position: 0.5; color: Qt.rgba(1,1,1,0.02) }
                            GradientStop { position: 1.0; color: "transparent" }
                        }
                    }

                    Rectangle {
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.bottom: parent.bottom
                        anchors.margins: 1
                        height: parent.height * 0.25
                        radius: 25
                        gradient: Gradient {
                            GradientStop { position: 0.0; color: "transparent" }
                            GradientStop { position: 1.0; color: glassShadow }
                        }
                    }
                }

                ChatPanel {
                    id: chatPanel
                    anchors.fill: parent
                    anchors.margins: 16
                    z: 2
                }
            }
        }

        // ───────────────────────────────────────────────────
        // RIGHT SIDE: COLLAPSIBLE STATUS CARDS
        // ───────────────────────────────────────────────────
        Item {
            id: rightPanel
            width: cardsCollapsed ? 54 : parent.width * 0.28 - 16
            height: parent.height

            Behavior on width {
                NumberAnimation { duration: 300; easing.type: Easing.InOutQuad }
            }

            // Collapse/Expand Button
            Rectangle {
                id: collapseButton
                anchors.left: parent.left
                anchors.top: parent.top
                width: 54
                height: 54
                radius: 27
                color: collapseMouseArea.containsMouse ? Qt.rgba(1, 1, 1, 0.15) : glassTint
                border.width: 1
                border.color: glassBorder
                z: 10

                Behavior on color {
                    ColorAnimation { duration: 150 }
                }

                Rectangle {
                    anchors.fill: parent
                    anchors.margins: 1
                    radius: 26
                    gradient: Gradient {
                        GradientStop { position: 0.0; color: glassHighlight }
                        GradientStop { position: 0.3; color: Qt.rgba(1,1,1,0.1) }
                        GradientStop { position: 1.0; color: "transparent" }
                    }
                }

                Column {
                    anchors.centerIn: parent
                    spacing: 4
                    rotation: cardsCollapsed ? 0 : 90
                    
                    Behavior on rotation {
                        NumberAnimation { duration: 300; easing.type: Easing.InOutQuad }
                    }
                    
                    Repeater {
                        model: 3
                        Rectangle {
                            width: 22
                            height: 3
                            radius: 1.5
                            color: Qt.rgba(1, 1, 1, 0.9)
                            scale: index === 1 ? 0.85 : 1.0
                        }
                    }
                }

                MouseArea {
                    id: collapseMouseArea
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: cardsCollapsed = !cardsCollapsed
                }
            }

            // Flickable INSIDE rightPanel
            Flickable {
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: collapseButton.bottom
                anchors.bottom: parent.bottom
                anchors.topMargin: 12
                clip: true
                visible: !cardsCollapsed
                opacity: cardsCollapsed ? 0 : 1
                contentHeight: cardsColumn.height
                boundsBehavior: Flickable.StopAtBounds

                Behavior on opacity {
                    NumberAnimation { duration: 200 }
                }
                // ===================
                //  CARDS COLUMN
                // ===================
                // STATUS CARDS (visible when expanded)
                Column {
                    id: cardsColumn
                    width: parent.width
                    spacing: 14

                    LiquidGlassCard {
                        width: parent.width
                        height: 140
                        title: "CPU LOAD"
                        value: window.cpuLoad
                    }

                    LiquidGlassCard {
                        width: parent.width
                        height: 140
                        title: "NETWORK"
                        value: window.networkOnline ? "ONLINE" : "OFFLINE"
                        statusIndicator: true
                        statusOn: window.networkOnline
                    }
                    // FILE BROWSER CARD
                    LiquidGlassCard {
                        width: parent.width
                        height: 140
                        title: "FILES"
                        value: "Browse"
                        
                        MouseArea {
                            anchors.fill: parent
                            cursorShape: Qt.PointingHandCursor
                            onClicked: fileBrowserPopup.open()
                        }
                    }

                    // Additional cards can be added here
                }
            }
        } 
    }

    // ═══════════════════════════════════════════════════════
    // WINDOW CONTROLS
    // ═══════════════════════════════════════════════════════
    Row {
        anchors.top: parent.top
        anchors.right: parent.right
        anchors.margins: 16
        spacing: 10
        z: 100

        Rectangle {
            width: 14; height: 14; radius: 7
            color: minMouse.containsMouse ? Qt.rgba(1, 0.85, 0.3, 1) : Qt.rgba(1, 0.8, 0.2, 0.9)
            border.width: 1
            border.color: Qt.rgba(0.8, 0.6, 0.1, 0.8)
            MouseArea {
                id: minMouse
                anchors.fill: parent
                hoverEnabled: true
                cursorShape: Qt.PointingHandCursor
                onClicked: window.showMinimized()
            }
        }

        Rectangle {
            width: 14; height: 14; radius: 7
            color: closeMouse.containsMouse ? Qt.rgba(1, 0.5, 0.5, 1) : Qt.rgba(1, 0.4, 0.4, 0.9)
            border.width: 1
            border.color: Qt.rgba(0.8, 0.2, 0.2, 0.8)
            MouseArea {
                id: closeMouse
                anchors.fill: parent
                hoverEnabled: true
                cursorShape: Qt.PointingHandCursor
                onClicked: Qt.quit()
            }
        }
    }
    
    // FILE BROWSER POPUP
    FileBrowserPopup {
        id: fileBrowserPopup
        x: (parent.width - width) / 2
        y: (parent.height - height) / 2
    }
}