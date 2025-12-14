import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    id: root
    Layout.fillWidth: true
    Layout.preferredHeight: 260

    // Exposed to MainView / backend
    property real audioLevel: 0.0  // 0–1

    // Internal state
    property real t: 0.0
    property real volEma: 0.0
    readonly property real alpha: 0.25

    // Geometry data
    property var particles: []
    property var rings: []
    property var arcs: []

    // Camera
    property real camDistance: 9.0
    property real orbitAngle: 0.0

    onAudioLevelChanged: {
        volEma = alpha * audioLevel + (1 - alpha) * volEma
    }

    Component.onCompleted: {
        generateParticles(300) // was 600
        generateRings(6)  // was 10
        generateArcs(8)    // was 12
    }

    function randomGaussian() {
        var u1 = Math.random(), u2 = Math.random()
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
    }

    function generateParticles(count) {
        var pts = []
        for (var i = 0; i < count; i++) {
            var x = randomGaussian(), y = randomGaussian(), z = randomGaussian()
            var len = Math.sqrt(x*x + y*y + z*z)
            if (len < 0.001) len = 1
            x /= len; y /= len; z /= len
            var r = 0.8 + Math.random() * 1.1
            pts.push({
                x: x * r, y: y * r, z: z * r,
                size: 0.02 + Math.random() * 0.04,
                alpha: 0.8
            })
        }
        particles = pts
    }

    function generateRings(count) {
        var r = []
        for (var i = 0; i < count; i++) {
            var radius = 1.3 + Math.random() * 1.0
            var ax = Math.random(), ay = Math.random(), az = Math.random()
            var alen = Math.sqrt(ax*ax + ay*ay + az*az)
            ax /= alen; ay /= alen; az /= alen
            var tiltAngle = Math.random() * Math.PI
            var pts = []
            for (var j = 0; j <= 100; j++) {
                var theta = (j / 100) * Math.PI * 2
                var px = radius * Math.cos(theta), py = radius * Math.sin(theta), pz = 0
                var rot = rotatePoint(px, py, pz, ax, ay, az, tiltAngle)
                pts.push(rot)
            }
            r.push({ points: pts, speed: 0.3 + 0.02 * i })
        }
        rings = r
    }

    function generateArcs(count) {
        var a = []
        for (var i = 0; i < count; i++) {
            var radius = 1.0 + Math.random() * 1.1
            var ax = Math.random(), ay = Math.random(), az = Math.random()
            var alen = Math.sqrt(ax*ax + ay*ay + az*az)
            ax /= alen; ay /= alen; az /= alen
            var tiltAngle = Math.random() * Math.PI * 2
            var pts = []
            for (var j = 0; j <= 50; j++) {
                var theta = (j / 50) * Math.PI
                var px = radius * Math.sin(theta), py = 0, pz = radius * Math.cos(theta)
                var rot = rotatePoint(px, py, pz, ax, ay, az, tiltAngle)
                pts.push(rot)
            }
            a.push({ points: pts, speed: 0.25 + 0.01 * i })
        }
        arcs = a
    }

    function rotatePoint(x, y, z, ax, ay, az, angle) {
        var ha = angle / 2, c = Math.cos(ha), s = Math.sin(ha)
        var qw = c, qx = -ax * s, qy = -ay * s, qz = -az * s
        var ix = qw*x + qy*z - qz*y
        var iy = qw*y + qz*x - qx*z
        var iz = qw*z + qx*y - qy*x
        var iw = -qx*x - qy*y - qz*z
        return {
            x: ix*qw + iw*(-qx) + iy*(-qz) - iz*(-qy),
            y: iy*qw + iw*(-qy) + iz*(-qx) - ix*(-qz),
            z: iz*qw + iw*(-qz) + ix*(-qy) - iy*(-qx)
        }
    }

    function rotateY(x, y, z, angle) {
        var c = Math.cos(angle), s = Math.sin(angle)
        return { x: x*c + z*s, y: y, z: -x*s + z*c }
    }

    function rotateX(x, y, z, angle) {
        var c = Math.cos(angle), s = Math.sin(angle)
        return { x: x, y: y*c - z*s, z: y*s + z*c }
    }

    function project(x, y, z, w, h) {
        var orbited = rotateY(x, y, z, orbitAngle * Math.PI / 180)
        var elevated = rotateX(orbited.x, orbited.y, orbited.z, 25 * Math.PI / 180)
        var fov = 300
        var depth = camDistance - elevated.z
        if (depth < 0.5) depth = 0.5
        var scale = fov / depth
        return {
            x: w/2 + elevated.x * scale,
            y: h/2 - elevated.y * scale,
            depth: depth,
            z: elevated.z
        }
    }

    Timer {
        interval: 32 // ~30 FPS
        running: true
        repeat: true
        onTriggered: {
            volEma = alpha * audioLevel + (1 - alpha) * volEma
            orbitAngle += 0.1
            t += 0.5
            canvas.requestPaint()
        }
    }

    // Transparent background - no black box
    Rectangle {
        anchors.fill: parent
        color: "transparent"
    }

    // Outer gimbal rings (QML elements for smooth rendering)
    Item {
        id: gimbalContainer
        anchors.centerIn: parent
        width: Math.min(parent.width, parent.height) - 40
        height: width

        // Outer horizontal ring
        Rectangle {
            id: outerRing
            anchors.centerIn: parent
            width: parent.width * 0.95
            height: 2
            radius: 1
            color: "transparent"
            border.width: 2
            border.color: Qt.rgba(1, 0.75, 0.2, 0.5)
            rotation: t * 0.3

            Repeater {
                model: 8
                Rectangle {
                    width: 6; height: 6; radius: 3
                    x: outerRing.width / 2 + (outerRing.width / 2 - 3) * Math.cos(index * Math.PI / 4) - 3
                    y: -2
                    color: Qt.rgba(1, 0.85, 0.3, 0.8)
                    rotation: -outerRing.rotation
                }
            }
        }

        // Inner vertical ring
        Rectangle {
            id: innerRing
            anchors.centerIn: parent
            width: 2
            height: parent.height * 0.85
            radius: 1
            color: "transparent"
            border.width: 2
            border.color: Qt.rgba(1, 0.8, 0.25, 0.4)
            rotation: -t * 0.4

            Repeater {
                model: 6
                Rectangle {
                    width: 5; height: 5; radius: 2.5
                    x: -1.5
                    y: innerRing.height / 2 + (innerRing.height / 2 - 2.5) * Math.cos(index * Math.PI / 3) - 2.5
                    color: Qt.rgba(1, 0.85, 0.3, 0.7)
                    rotation: -innerRing.rotation
                }
            }
        }
    }

    Canvas {
        id: canvas
        anchors.fill: parent
        renderStrategy: Canvas.Threaded

        onPaint: {
            var ctx = getContext("2d")
            var w = width, h = height
            ctx.clearRect(0, 0, w, h)

            var vol = Math.min(1.0, volEma * 12.0)
            var coreScale = 1.0 + 0.4 * Math.sin(t / 15.0) + 0.3 * vol
            coreScale = Math.max(0.7, Math.min(2.0, coreScale))

            // Particles
            var projP = []
            var pRot = t * 0.1 * Math.PI / 180
            for (var i = 0; i < particles.length; i++) {
                var p = particles[i]
                var r = rotateY(p.x, p.y, p.z, pRot)
                var pr = project(r.x, r.y, r.z, w, h)
                projP.push({
                    x: pr.x, y: pr.y,
                    size: p.size * 300 / pr.depth,
                    alpha: p.alpha * Math.max(0, 1 - pr.z / 3),
                    depth: pr.depth
                })
            }
            projP.sort(function(a, b) { return b.depth - a.depth })

            for (var i = 0; i < projP.length; i++) {
                var pp = projP[i]
                if (pp.alpha > 0.08 && pp.size > 0.4) {
                    ctx.beginPath()
                    ctx.arc(pp.x, pp.y, pp.size, 0, Math.PI * 2)
                    ctx.fillStyle = "rgba(255, 200, 50, " + (pp.alpha * 0.7) + ")"
                    ctx.fill()
                }
            }

            // Rings
            for (var i = 0; i < rings.length; i++) {
                var ring = rings[i]
                var rRot = t * ring.speed * Math.PI / 180
                var rAlpha = 0.18 + 0.12 * Math.sin(t / 20.0 + i)
                ctx.beginPath()
                ctx.strokeStyle = "rgba(255, 199, 38, " + rAlpha + ")"
                ctx.lineWidth = 1
                var first = true
                for (var j = 0; j < ring.points.length; j++) {
                    var pt = ring.points[j]
                    var r = rotateY(pt.x, pt.y, pt.z, rRot)
                    var pr = project(r.x, r.y, r.z, w, h)
                    if (first) { ctx.moveTo(pr.x, pr.y); first = false }
                    else { ctx.lineTo(pr.x, pr.y) }
                }
                ctx.stroke()
            }

            // Arcs
            for (var i = 0; i < arcs.length; i++) {
                var arc = arcs[i]
                var aRot = t * arc.speed * Math.PI / 180
                var aAlpha = 0.1 + 0.12 * Math.sin(t / 25.0 + i)
                ctx.beginPath()
                ctx.strokeStyle = "rgba(255, 179, 51, " + aAlpha + ")"
                ctx.lineWidth = 0.8
                var first = true
                for (var j = 0; j < arc.points.length; j++) {
                    var pt = arc.points[j]
                    var r = rotateX(pt.x, pt.y, pt.z, aRot)
                    var pr = project(r.x, r.y, r.z, w, h)
                    if (first) { ctx.moveTo(pr.x, pr.y); first = false }
                    else { ctx.lineTo(pr.x, pr.y) }
                }
                ctx.stroke()
            }

            // Core glow - softer and more natural
            var cp = project(0, 0, 0, w, h)
            var baseR = 0.3 * 300 / cp.depth * coreScale

            var g = ctx.createRadialGradient(cp.x, cp.y, 0, cp.x, cp.y, baseR * 5)
            g.addColorStop(0, "rgba(255, 210, 80, 0.35)")
            g.addColorStop(0.3, "rgba(255, 180, 50, 0.12)")
            g.addColorStop(1, "rgba(255, 150, 0, 0)")
            ctx.beginPath()
            ctx.arc(cp.x, cp.y, baseR * 5, 0, Math.PI * 2)
            ctx.fillStyle = g
            ctx.fill()

            g = ctx.createRadialGradient(cp.x, cp.y, 0, cp.x, cp.y, baseR * 2.5)
            g.addColorStop(0, "rgba(255, 220, 120, 0.5)")
            g.addColorStop(0.5, "rgba(255, 190, 60, 0.2)")
            g.addColorStop(1, "rgba(255, 160, 30, 0)")
            ctx.beginPath()
            ctx.arc(cp.x, cp.y, baseR * 2.5, 0, Math.PI * 2)
            ctx.fillStyle = g
            ctx.fill()

            g = ctx.createRadialGradient(cp.x, cp.y, 0, cp.x, cp.y, baseR)
            g.addColorStop(0, "rgba(255, 245, 200, 0.85)")
            g.addColorStop(0.4, "rgba(255, 210, 80, 0.6)")
            g.addColorStop(1, "rgba(255, 170, 40, 0.3)")
            ctx.beginPath()
            ctx.arc(cp.x, cp.y, baseR, 0, Math.PI * 2)
            ctx.fillStyle = g
            ctx.fill()
        }
    }

    // Status bar
    RowLayout {
        anchors {
            left: parent.left
            right: parent.right
            bottom: parent.bottom
            margins: 16
        }
        spacing: 12

        Rectangle {
            Layout.preferredWidth: statusLabel.implicitWidth + 24
            Layout.preferredHeight: 30
            radius: 15
            color: Qt.rgba(0.12, 0.1, 0.05, 0.6)
            border.width: 1
            border.color: audioLevel > 0.04 
                ? Qt.rgba(1, 0.8, 0.3, 0.8)
                : Qt.rgba(0.6, 0.5, 0.2, 0.4)

            Row {
                anchors.centerIn: parent
                spacing: 6

                Rectangle {
                    width: 6; height: 6; radius: 3
                    anchors.verticalCenter: parent.verticalCenter
                    color: audioLevel > 0.04 ? "#ffcc33" : "#806622"

                    SequentialAnimation on scale {
                        running: audioLevel > 0.04
                        loops: Animation.Infinite
                        NumberAnimation { to: 1.3; duration: 600 }
                        NumberAnimation { to: 1.0; duration: 600 }
                    }
                }

                Text {
                    id: statusLabel
                    text: audioLevel > 0.04 ? "ACTIVE" : "STANDBY"
                    font.pixelSize: 10
                    font.letterSpacing: 1.5
                    font.weight: Font.Bold
                    color: Qt.rgba(1, 0.85, 0.4, 0.9)
                }
            }
        }

        Item {
            Layout.fillWidth: true
            Layout.preferredHeight: 6

            Rectangle {
                anchors.fill: parent
                radius: 3
                color: Qt.rgba(0.15, 0.12, 0.05, 0.5)
                border.width: 1
                border.color: Qt.rgba(0.5, 0.4, 0.2, 0.3)
            }

            Rectangle {
                width: parent.width * Math.min(1, volEma * 2.5)
                height: parent.height
                radius: 3
                gradient: Gradient {
                    orientation: Gradient.Horizontal
                    GradientStop { position: 0.0; color: Qt.rgba(0.8, 0.6, 0.1, 0.8) }
                    GradientStop { position: 0.7; color: Qt.rgba(1, 0.8, 0.2, 0.9) }
                    GradientStop { position: 1.0; color: Qt.rgba(1, 0.9, 0.4, 1) }
                }
                Behavior on width { NumberAnimation { duration: 100 } }
            }
        }

        Rectangle {
            Layout.preferredWidth: 50
            Layout.preferredHeight: 30
            radius: 15
            color: Qt.rgba(0.12, 0.1, 0.05, 0.6)
            border.width: 1
            border.color: Qt.rgba(0.5, 0.4, 0.2, 0.3)

            Text {
                anchors.centerIn: parent
                text: qsTr("%1%").arg(Math.round(Math.min(100, volEma * 250)))
                font.pixelSize: 11
                font.weight: Font.Bold
                font.family: "Courier"
                color: Qt.rgba(1, 0.85, 0.4, 0.9)
            }
        }
    }
}