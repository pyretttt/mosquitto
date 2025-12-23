import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: win
    visible: true
    title: "PySide6 QML - Redux Sidebars"

    RowLayout {
        anchors.fill: parent
        spacing: 0

        // LEFT SIDEBAR (same cell type, different heights)
        Rectangle {
            Layout.preferredWidth: win.width / 5
            Layout.fillHeight: true
            color: "#1e1e1e"
            border.width: 1
            border.color: "#333333"

            ListView {
                id: leftList
                anchors.fill: parent
                clip: true
                model: store.leftModel
                spacing: 0

                delegate: Rectangle {
                    id: cell
                    width: leftList.width

                    // self-size based on contents
                    implicitHeight: content.implicitHeight + 20
                    height: implicitHeight

                    color: {
                        if (isSelected)
                            return "#343333ff";
                        return (index % 2 === 0) ? "#232323" : "#1f1f1f";
                    }

                    Column {
                        id: content
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.margins: 10
                        spacing: 6

                        Text {
                            text: name
                            color: "white"
                            font.bold: true
                            wrapMode: Text.WordWrap
                            width: parent.width
                        }

                        Text {
                            text: description
                            color: "#cfcfcf"
                            wrapMode: Text.WordWrap
                            width: parent.width
                            visible: description.length > 0
                        }
                    }
                    TapHandler {
                        onTapped: {
                            store.dispatch({"type": "LEFT_ITEM_TAPPED", "payload": {"index": index}});
                        }
                        onPressedChanged: cell.opacity = pressed ? 0.7 : 1.0
                    }
                }

                ScrollBar.vertical: ScrollBar { }
            }
        }

        // CENTER (just a colored background)
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: store.centerColor

            Text {
                anchors.centerIn: parent
                text: "CENTER"
                color: "#bbbbbb"
                font.pixelSize: 26
            }
        }

        // RIGHT SIDEBAR (different cell types, different heights)
        Rectangle {
            Layout.preferredWidth: win.width / 5
            Layout.fillHeight: true
            color: "#141414"
            border.width: 1
            border.color: "#333333"

            ListView {
                id: rightList
                anchors.fill: parent
                clip: true
                model: store.rightModel
                spacing: 12
                topMargin: 12
                bottomMargin: 12
                leftMargin: 12
                rightMargin: 12

                delegate: Item {
                    width: rightList.width - rightList.leftMargin - rightList.rightMargin
                    implicitHeight: loader.implicitHeight
                    height: implicitHeight


                    // Choose component by `type` role
                    Loader {
                        id: loader
                        anchors.left: parent.left
                        anchors.right: parent.right
                        sourceComponent: {
                            if (type === "header") return headerComp
                            if (type === "toggle") return toggleComp
                            if (type === "button") return buttonComp
                            return cardComp
                        }
                    }

                    // --- Components ---
                    Component {
                        id: headerComp
                        Item {
                            implicitHeight: headerText.implicitHeight
                            Text {
                                id: headerText
                                width: parent.width
                                text: model.text
                                color: "white"
                                font.pixelSize: 18
                                font.bold: true
                                wrapMode: Text.WordWrap
                            }
                        }
                    }


                    Component {
                        id: cardComp
                        Rectangle {
                            radius: 10
                            color: "#1f1f1f"
                            border.width: 1
                            border.color: "#2d2d2d"
                            implicitHeight: col.implicitHeight + 18

                            Column {
                                id: col
                                anchors.fill: parent
                                anchors.margins: 10
                                spacing: 6

                                Text {
                                    width: parent.width
                                    text: title
                                    color: "white"
                                    font.bold: true
                                    wrapMode: Text.WordWrap
                                }
                                Text {
                                    width: parent.width
                                    text: body
                                    color: "#cfcfcf"
                                    wrapMode: Text.WordWrap
                                }
                            }
                        }
                    }

                    Component {
                        id: toggleComp
                        Rectangle {
                            radius: 10
                            color: "#1a1a1a"
                            border.width: 1
                            border.color: "#2d2d2d"
                            implicitHeight: 56

                            RowLayout {
                                anchors.fill: parent
                                anchors.margins: 10
                                spacing: 10

                                Text {
                                    Layout.fillWidth: true
                                    text: model.text
                                    color: "white"
                                    wrapMode: Text.WordWrap
                                }

                                Switch {
                                    checked: model.checked
                                    onToggled: store.dispatch({ "type": "TOGGLE_RIGHT", "payload": { "row": index } })
                                }
                            }
                        }
                    }

                    Component {
                        id: buttonComp
                        Item {
                            implicitHeight: btn.implicitHeight
                            Button {
                                id: btn
                                anchors.left: parent.left
                                anchors.right: parent.right
                                text: model.text

                                onClicked: {
                                    if (actionType === "SET_CENTER_COLOR") {
                                        store.dispatch({ "type": "SET_CENTER_COLOR", "payload": model.payload || {} })
                                    } else if (actionType === "ADD_LEFT") {
                                        store.dispatch({ "type": "ADD_LEFT", "payload": { "name": "Added", "description": "Created from a right-sidebar button" } })
                                    } else {
                                        store.dispatch({ "type": actionType })
                                    }
                                }
                            }
                        }
                    }
                }

                ScrollBar.vertical: ScrollBar { }
            }
        }
    }
}