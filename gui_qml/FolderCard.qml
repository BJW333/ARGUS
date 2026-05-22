import QtQuick 2.15
import QtQuick.Controls 2.15

Item {
    id: root
    width: 140
    height: 100

    property string label: "Folder"
    property var contents: []
    property bool isHovered: false

    signal clicked()
    signal cardDropped(var card)

    Rectangle {
        id: bg
        anchors.fill: parent
        radius: 18
        color: Qt.rgba(1, 1, 1, 0.08)
        border.width: 1
        border.color: isHovered ? Qt.rgba(1, 0.8, 0.3, 0.6) : Qt.rgba(1, 1, 1, 0.2)
        scale: isHovered ? 1.03 : 1.0

        Behavior on scale {
            NumberAnimation { duration: 150; easing.type: Easing.OutQuad }
        }
        Behavior on border.color {
            ColorAnimation { duration: 150 }
        }

        // Highlight
        Rectangle {
            anchors.fill: parent
            anchors.margins: 1
            radius: 17
            gradient: Gradient {
                GradientStop { position: 0.0; color: Qt.rgba(1,1,1,0.2) }
                GradientStop { position: 0.4; color: Qt.rgba(1,1,1,0.05) }
                GradientStop { position: 1.0; color: "transparent" }
            }
        }
    }

    Column {
        anchors.centerIn: parent
        spacing: 8

        Text {
            text: "📁"
            font.pixelSize: 28
            anchors.horizontalCenter: parent.horizontalCenter
        }

        Text {
            text: root.label
            color: Qt.rgba(1, 0.85, 0.4, 0.9)
            font.pixelSize: 12
            font.weight: Font.Bold
            anchors.horizontalCenter: parent.horizontalCenter
        }
    }

    // Badge
    Rectangle {
        visible: contents.length > 0
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.margins: 8
        width: 22; height: 22; radius: 11
        color: Qt.rgba(1, 0.3, 0.4, 0.9)

        Text {
            anchors.centerIn: parent
            text: contents.length
            color: "white"
            font.pixelSize: 11
            font.weight: Font.Bold
        }
    }

    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        cursorShape: Qt.PointingHandCursor
        onEntered: isHovered = true
        onExited: isHovered = false
        onClicked: root.clicked()
    }

    function absorb(item) {
        contents.push(item)
        contentsChanged()
    }

    function release(item) {
        var idx = contents.indexOf(item)
        if (idx >= 0) {
            contents.splice(idx, 1)
            contentsChanged()
        }
    }
}
