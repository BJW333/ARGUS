import QtQuick 2.15
import QtQuick.Controls 2.15
import "."

Item {
    id: root

    // Public properties
    property string title: "METRIC"
    property string value: "0%"
    property bool statusIndicator: false
    property bool statusOn: true
    property color accentColor: Qt.rgba(0.4, 0.7, 1.0, 0.5)

    // Glass design tokens
    readonly property real glassOpacity: 0.10
    readonly property color glassTint: Qt.rgba(1, 1, 1, glassOpacity)
    readonly property color glassBorder: Qt.rgba(1, 1, 1, 0.08)
    readonly property color glassHighlight: Qt.rgba(1, 1, 1, 0.38)

    // Hover state
    property bool isHovered: false

    Rectangle {
        id: cardGlass
        anchors.fill: parent
        radius: 22
        color: glassTint
        border.width: 1
        border.color: isHovered ? Qt.rgba(1,1,1,0.15) : glassBorder

        scale: isHovered ? 1.02 : 1.0
        Behavior on scale {
            NumberAnimation { duration: 180; easing.type: Easing.OutQuad }
        }
        Behavior on border.color {
            ColorAnimation { duration: 150 }
        }

        // Top specular highlight
        Rectangle {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.margins: 1
            height: parent.height * 0.55
            radius: 21
            gradient: Gradient {
                GradientStop { position: 0.0; color: glassHighlight }
                GradientStop { position: 0.2; color: Qt.rgba(1,1,1,0.15) }
                GradientStop { position: 0.5; color: Qt.rgba(1,1,1,0.04) }
                GradientStop { position: 1.0; color: "transparent" }
            }
        }

        // Bottom shadow gradient
        Rectangle {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            anchors.margins: 1
            height: parent.height * 0.35
            radius: 21
            gradient: Gradient {
                GradientStop { position: 0.0; color: "transparent" }
                GradientStop { position: 0.6; color: Qt.rgba(0,0,0,0.08) }
                GradientStop { position: 1.0; color: Qt.rgba(0,0,0,0.2) }
            }
        }

        // Accent glow (simple border, no shader)
        // Rectangle {
        //     anchors.fill: parent
        //     anchors.margins: -2
        //     radius: 24
        //     color: "transparent"
        //     border.width: 2
        //     border.color: accentColor
        //     opacity: isHovered ? 0.6 : 0.25
        //     Behavior on opacity {
        //         NumberAnimation { duration: 200 }
        //     }
        // }
    }

    // Content
    Column {
        anchors.centerIn: parent
        spacing: 10

        // Title
        Text {
            text: root.title
            color: Qt.rgba(1, 1, 1, 0.65)
            font.pixelSize: 11
            font.letterSpacing: 1.2
            font.weight: Font.Medium
            font.capitalization: Font.AllUppercase
            anchors.horizontalCenter: parent.horizontalCenter
        }

        // Status indicator (no Glow shader)
        Rectangle {
            visible: root.statusIndicator
            width: 12
            height: 12
            radius: 6
            anchors.horizontalCenter: parent.horizontalCenter
            color: root.statusOn ? "#00FF88" : "#FF4466"

            // Simple glow approximation with layered circles
            Rectangle {
                anchors.centerIn: parent
                width: 20
                height: 20
                radius: 10
                color: root.statusOn ? Qt.rgba(0, 1, 0.53, 0.3) : Qt.rgba(1, 0.27, 0.4, 0.3)
                z: -1
            }
            Rectangle {
                anchors.centerIn: parent
                width: 28
                height: 28
                radius: 14
                color: root.statusOn ? Qt.rgba(0, 1, 0.53, 0.15) : Qt.rgba(1, 0.27, 0.4, 0.15)
                z: -2
            }

            // Pulse animation when online
            SequentialAnimation on opacity {
                running: root.statusOn && root.statusIndicator
                loops: Animation.Infinite
                NumberAnimation { to: 0.6; duration: 800; easing.type: Easing.InOutQuad }
                NumberAnimation { to: 1.0; duration: 800; easing.type: Easing.InOutQuad }
            }
        }

        // Value (no text glow)
        Text {
            text: root.value
            color: Qt.rgba(1, 1, 1, 0.98)
            font.pixelSize: root.statusIndicator ? 16 : 28
            font.weight: Font.Bold
            anchors.horizontalCenter: parent.horizontalCenter
        }
    }

    // Hover detection
    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        onEntered: root.isHovered = true
        onExited: root.isHovered = false
    }
}