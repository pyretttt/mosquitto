import QtQuick
import QtQuick.Controls

ApplicationWindow {
    width: 420
    height: 360
    visible: true

    ListView {
        anchors.fill: parent
        model: leftDataSideBar.property   // <- list of maps

        delegate: Item {
            width: ListView.view.width
            height: 64

            Column {
                anchors.verticalCenter: parent.verticalCenter
                anchors.left: parent.left
                anchors.leftMargin: 12
                spacing: 2

                Text { text: "name"; font.pixelSize: 16 }      // keys become roles
                Text { text: description; opacity: 0.7 }
            }

            MouseArea {
                anchors.fill: parent
                onClicked: console.log("Clicked:", id)
            }
        }
    }
}