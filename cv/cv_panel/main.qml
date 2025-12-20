import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window
    visible: true
    width: 420
    height: 640
    title: ""

    property QtObject leftSideBarData: null
    property var itemsModel: []

    Connections {
        target: window.leftSideBarData
        function onSignal(data) {
            window.itemsModel = data;
        }
    }

    ListView {
        id: list
        anchors.fill: parent
        anchors.margins: 12
        spacing: 10
        clip: true

        model: window.itemsModel


        delegate: Rectangle {
            id: card
            width: list.width
            radius: 12
            color: "#f2f2f2"
            border.color: "#d0d0d0"

            // self-sizing height driven by content
            height: content.implicitHeight + 24

            ColumnLayout {
                id: content
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.margins: 12
                spacing: 8

                ColumnLayout {
                    Layout.fillWidth: true

                    Label {
                        text: modelData.name
                        font.bold: true
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }

                    Label {
                        text: modelData.description
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }
                }

                Button {
                    text: "Tap"
                    onClicked: console.log("Tapped item:", modelData.name)
                }
            }
        }
    }
}
