import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "."

Item {
    id: root

    ListModel { id: messagesModel }

    // ═══════════════════════════════════════════════════════
    // MESSAGE LIST
    // ═══════════════════════════════════════════════════════
    ListView {
        id: listView
        anchors {
            top: parent.top
            left: parent.left
            right: parent.right
            bottom: inputContainer.top
            margins: 12
            bottomMargin: 14
        }
        clip: true
        spacing: 10
        boundsBehavior: Flickable.StopAtBounds
        model: messagesModel
        verticalLayoutDirection: ListView.TopToBottom
        
        
        property bool stickToBottom: true
        property int bottomThreshold: 40
        function nearBottom() {
            return (contentY + height) >= (contentHeight - bottomThreshold)
        }

        // if user scrolls up, stop autoscroll; if they scroll back down, re-enable
        onContentYChanged: stickToBottom = nearBottom()
        
        onContentHeightChanged: {
            if (stickToBottom) Qt.callLater(function(){ positionViewAtEnd() })
        }

        // Custom scrollbar
        ScrollBar.vertical: ScrollBar {
            width: 6
            policy: ScrollBar.AsNeeded
            contentItem: Rectangle {
                radius: 3
                color: Qt.rgba(1, 1, 1, 0.25)
            }
        }

        delegate: Item {
            width: ListView.view.width
            height: bubble.height + 6

            property bool isUser: role === "user" || role === "me"
            property bool isAssistant: role === "assistant"
            property bool isSystem: role === "system"

            Row {
                anchors.fill: parent
                anchors.leftMargin: 6
                anchors.rightMargin: 6
                layoutDirection: isUser ? Qt.RightToLeft : Qt.LeftToRight
                spacing: 8

                // Chat bubble
                Rectangle {
                    id: bubble
                    width: Math.min(textItem.implicitWidth + 28, parent.width * 0.78)
                    height: textItem.implicitHeight + 18
                    radius: 18

                    // Different glass tints per role
                    color: isUser
                        ? Qt.rgba(1.0, 0.8, 0.3, 0.18)
                        : (isSystem
                            ? Qt.rgba(1, 1, 1, 0.06)
                            : Qt.rgba(1, 1, 1, 0.12))

                    border.width: 1
                    border.color: isUser
                        ? Qt.rgba(1.0, 0.85, 0.4, 0.45)
                        : Qt.rgba(1, 1, 1, 0.22)

                    // Top highlight
                    Rectangle {
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.margins: 1
                        height: parent.height * 0.5
                        radius: 17
                        gradient: Gradient {
                            GradientStop { position: 0.0; color: Qt.rgba(1,1,1,0.2) }
                            GradientStop { position: 0.4; color: Qt.rgba(1,1,1,0.06) }
                            GradientStop { position: 1.0; color: "transparent" }
                        }
                    }

                    // Message text
                    Text {
                        id: textItem
                        anchors.fill: parent
                        anchors.margins: 12
                        text: model.text
                        wrapMode: Text.Wrap
                        color: isUser
                            ? Qt.rgba(1, 1, 1, 1)
                            : (isSystem 
                                ? Qt.rgba(1, 1, 1, 0.6)
                                : Qt.rgba(1, 1, 1, 0.95))
                        font.pixelSize: isSystem ? 12 : 14
                        font.italic: isSystem
                        lineHeight: 1.4
                    }

                    // Entry animation
                    scale: 0.9
                    opacity: 0
                    Component.onCompleted: {
                        scale = 1.0
                        opacity = 1.0
                    }
                    Behavior on scale {
                        NumberAnimation { duration: 200; easing.type: Easing.OutBack }
                    }
                    Behavior on opacity {
                        NumberAnimation { duration: 150 }
                    }
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════
    // INPUT BAR
    // ═══════════════════════════════════════════════════════
    Rectangle {
        id: inputContainer
        anchors {
            left: parent.left
            right: parent.right
            bottom: parent.bottom
            leftMargin: 12
            rightMargin: 12
            bottomMargin: 12
        }
        height: 50
        radius: 25
        color: Qt.rgba(1, 1, 1, 0.08)
        border.width: 1
        border.color: inputFocused 
            ? Qt.rgba(1.0, 0.8, 0.3, 0.5)
            : Qt.rgba(1, 1, 1, 0.2)

        property bool inputFocused: input.activeFocus

        Behavior on border.color {
            ColorAnimation { duration: 150 }
        }

        // Top specular
        Rectangle {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.margins: 1
            height: parent.height * 0.5
            radius: 24
            gradient: Gradient {
                GradientStop { position: 0.0; color: Qt.rgba(1,1,1,0.18) }
                GradientStop { position: 0.5; color: Qt.rgba(1,1,1,0.04) }
                GradientStop { position: 1.0; color: "transparent" }
            }
        }

        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 18
            anchors.rightMargin: 8
            spacing: 10

            // Text input
            TextInput {
                id: input
                Layout.fillWidth: true
                Layout.fillHeight: true
                verticalAlignment: Text.AlignVCenter
                color: Qt.rgba(1, 1, 1, 0.95)
                font.pixelSize: 14
                clip: true
                selectByMouse: true
                selectionColor: Qt.rgba(0.4, 0.7, 1.0, 0.4)

                Keys.onReturnPressed: (event) => {
                    if (!event.isAutoRepeat) {
                        send()
                        event.accepted = true
                    }
                }

                // Placeholder
                Text {
                    anchors.fill: parent
                    visible: input.text.length === 0 && !input.activeFocus
                    text: "Message ARGUS..."
                    color: Qt.rgba(1, 1, 1, 0.4)
                    font.pixelSize: 14
                    verticalAlignment: Text.AlignVCenter
                }
            }

            // Send button
            Rectangle {
                id: sendButton
                Layout.preferredWidth: 72
                Layout.preferredHeight: 36
                radius: 18
                
                property bool hovered: sendMouse.containsMouse
                property bool pressed: sendMouse.containsPress

                color: pressed
                    ? Qt.rgba(1.0, 0.7, 0.2, 0.6)
                    : (hovered
                        ? Qt.rgba(1.0, 0.8, 0.3, 0.45)
                        : Qt.rgba(1.0, 0.8, 0.3, 0.3))

                border.width: 1
                border.color: Qt.rgba(1.0, 0.85, 0.4, hovered ? 0.8 : 0.5)

                Behavior on color {
                    ColorAnimation { duration: 100 }
                }

                Behavior on border.color {
                    ColorAnimation { duration: 100 }
                }

                // Button highlight
                Rectangle {
                    anchors.fill: parent
                    anchors.margins: 1
                    radius: 17
                    gradient: Gradient {
                        GradientStop { position: 0.0; color: Qt.rgba(1,1,1,0.25) }
                        GradientStop { position: 0.5; color: Qt.rgba(1,1,1,0.05) }
                        GradientStop { position: 1.0; color: "transparent" }
                    }
                }

                Text {
                    anchors.centerIn: parent
                    text: "Send"
                    color: Qt.rgba(1, 1, 1, 0.95)
                    font.pixelSize: 13
                    font.weight: Font.Medium
                }

                MouseArea {
                    id: sendMouse
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: send()
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════
    // FUNCTIONS
    // ═══════════════════════════════════════════════════════
    function send() {
        var t = input.text.trim()
        if (t.length === 0) return
        Backend.sendMessage(t)
        input.text = ""
        input.forceActiveFocus()
    }

    Component.onCompleted: {
        Backend.messageReceived.connect(function(role, text) {
            messagesModel.append({ "role": role, "text": text })
            if (listView.stickToBottom)
                Qt.callLater(function(){ listView.positionViewAtEnd() })
        })
    }
}