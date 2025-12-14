import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Popup {
    id: fileBrowser
    width: 500
    height: 450
    modal: true
    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
    
    property string currentPath: Backend.getHomePath()
    property var breadcrumbs: [currentPath]
    property string searchQuery: ""
    property var fileList: []

    background: Rectangle {
        color: Qt.rgba(0.12, 0.1, 0.08, 0.95)
        radius: 22
        border.width: 1
        border.color: Qt.rgba(1, 0.8, 0.3, 0.3)

        Rectangle {
            anchors.fill: parent
            anchors.margins: 1
            radius: 21
            gradient: Gradient {
                GradientStop { position: 0.0; color: Qt.rgba(1, 0.9, 0.7, 0.15) }
                GradientStop { position: 0.3; color: Qt.rgba(1, 0.85, 0.5, 0.05) }
                GradientStop { position: 1.0; color: "transparent" }
            }
        }
    }

    onOpened: refresh()

    function refresh() {
        if (searchQuery.length > 0) {
            fileList = Backend.searchFiles(currentPath, searchQuery)
        } else {
            fileList = Backend.listDirectory(currentPath)
        }
    }

    function navigateTo(path) {
        var idx = breadcrumbs.indexOf(path)
        if (idx >= 0) {
            breadcrumbs = breadcrumbs.slice(0, idx + 1)
        } else {
            breadcrumbs = breadcrumbs.concat([path])
        }
        currentPath = path
        searchQuery = ""
        searchInput.text = ""
        refresh()
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 12

        // Title bar
        RowLayout {
            Layout.fillWidth: true

            Text {
                text: "📁 Files"
                color: Qt.rgba(1, 0.85, 0.4, 0.95)
                font.pixelSize: 16
                font.weight: Font.Bold
            }

            Item { Layout.fillWidth: true }

            Rectangle {
                width: 28; height: 28; radius: 14
                color: closeArea.containsMouse ? Qt.rgba(1, 0.4, 0.3, 0.8) : Qt.rgba(1, 1, 1, 0.1)
                
                Text {
                    anchors.centerIn: parent
                    text: "✕"
                    color: Qt.rgba(1, 0.9, 0.8, 0.9)
                    font.pixelSize: 12
                }
                MouseArea {
                    id: closeArea
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: fileBrowser.close()
                }
            }
        }

        // Breadcrumbs
        ScrollView {
            Layout.fillWidth: true
            Layout.preferredHeight: 32
            clip: true

            Row {
                spacing: 4
                Repeater {
                    model: breadcrumbs
                    Row {
                        spacing: 4
                        Rectangle {
                            width: crumbText.implicitWidth + 16
                            height: 26
                            radius: 13
                            color: crumbMouse.containsMouse ? Qt.rgba(1, 0.8, 0.3, 0.25) : Qt.rgba(1, 0.8, 0.3, 0.12)
                            
                            Text {
                                id: crumbText
                                anchors.centerIn: parent
                                text: {
                                    var parts = modelData.split("/")
                                    return parts[parts.length - 1] || "/"
                                }
                                color: Qt.rgba(1, 0.85, 0.4, 0.95)
                                font.pixelSize: 11
                                font.weight: Font.Medium
                            }
                            MouseArea {
                                id: crumbMouse
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onClicked: navigateTo(modelData)
                            }
                        }
                        Text {
                            text: "›"
                            color: Qt.rgba(1, 0.8, 0.3, 0.5)
                            font.pixelSize: 14
                            visible: index < breadcrumbs.length - 1
                            anchors.verticalCenter: parent.verticalCenter
                        }
                    }
                }
            }
        }

        // Search bar
        Rectangle {
            Layout.fillWidth: true
            height: 40
            radius: 20
            color: Qt.rgba(1, 1, 1, 0.06)
            border.width: 1
            border.color: searchInput.activeFocus 
                ? Qt.rgba(1, 0.8, 0.3, 0.6) 
                : Qt.rgba(1, 1, 1, 0.12)

            Behavior on border.color {
                ColorAnimation { duration: 150 }
            }

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 14
                anchors.rightMargin: 14
                spacing: 8

                Text {
                    text: "🔍"
                    font.pixelSize: 14
                    opacity: 0.7
                }

                TextInput {
                    id: searchInput
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    verticalAlignment: Text.AlignVCenter
                    color: Qt.rgba(1, 0.95, 0.9, 0.95)
                    font.pixelSize: 13
                    clip: true
                    selectByMouse: true
                    selectionColor: Qt.rgba(1, 0.8, 0.3, 0.4)

                    onTextChanged: {
                        searchQuery = text
                        searchTimer.restart()
                    }

                    Text {
                        anchors.fill: parent
                        verticalAlignment: Text.AlignVCenter
                        text: "Search files..."
                        color: Qt.rgba(1, 0.9, 0.7, 0.35)
                        font.pixelSize: 13
                        visible: searchInput.text.length === 0
                    }
                }

                Rectangle {
                    width: 20; height: 20; radius: 10
                    color: clearMouse.containsMouse ? Qt.rgba(1, 0.8, 0.3, 0.3) : "transparent"
                    visible: searchInput.text.length > 0

                    Text {
                        anchors.centerIn: parent
                        text: "✕"
                        color: Qt.rgba(1, 0.9, 0.7, 0.6)
                        font.pixelSize: 10
                    }
                    MouseArea {
                        id: clearMouse
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: {
                            searchInput.text = ""
                            refresh()
                        }
                    }
                }
            }
        }

        Timer {
            id: searchTimer
            interval: 300
            onTriggered: refresh()
        }

        // File list
        ListView {
            id: fileListView
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 6
            model: fileList
            boundsBehavior: Flickable.StopAtBounds

            ScrollBar.vertical: ScrollBar {
                width: 6
                policy: ScrollBar.AsNeeded
                contentItem: Rectangle {
                    radius: 3
                    color: Qt.rgba(1, 0.8, 0.3, 0.4)
                }
            }

            delegate: Rectangle {
                width: fileListView.width
                height: 44
                radius: 12
                color: fileMouse.containsMouse 
                    ? Qt.rgba(1, 0.8, 0.3, 0.15) 
                    : Qt.rgba(1, 1, 1, 0.04)

                Behavior on color {
                    ColorAnimation { duration: 100 }
                }

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 14
                    anchors.rightMargin: 14
                    spacing: 10

                    Text {
                        text: modelData.isDir ? "📁" : "📄"
                        font.pixelSize: 18
                    }

                    Text {
                        Layout.fillWidth: true
                        text: modelData.name
                        color: Qt.rgba(1, 0.95, 0.9, 0.9)
                        font.pixelSize: 13
                        elide: Text.ElideMiddle
                    }

                    Text {
                        text: modelData.isDir ? "›" : ""
                        color: Qt.rgba(1, 0.8, 0.3, 0.5)
                        font.pixelSize: 16
                    }
                }

                MouseArea {
                    id: fileMouse
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: {
                        if (modelData.isDir) {
                            navigateTo(modelData.path)
                        } else {
                            Backend.openFile(modelData.path)
                        }
                    }
                }
            }

            Text {
                anchors.centerIn: parent
                text: fileList.length === 0 ? "No files found" : ""
                color: Qt.rgba(1, 0.85, 0.5, 0.4)
                font.pixelSize: 14
            }
        }
    }
}